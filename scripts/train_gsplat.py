#!/usr/bin/env python3
"""train_gsplat.py — gsplat 학습 (포즈 고정 + dense depth supervision). Phase 5 (06_train_gsplat.sh가 호출).

입력 : processed/<scene>/colmap/ (cameras/images=Tcw/points3D + rgb + depth via associations)
출력 : outputs/<scene>/gsplat/{scene.ply, train_log.jsonl, renders/}

- **포즈 고정 증거**: 옵티마이저 param group에 카메라 포즈 텐서가 없다(means/scales/quats/opacities/sh0/shN만).
  gsplat은 viewmats를 상수로 받으므로 포즈는 학습되지 않음(decoupled 설계).
- **depth supervision**: rasterization(render_mode="RGB+ED")로 expected depth 렌더 → 센서 depth와 L1(depth>0 픽셀).
- 기본 손실: (1-λs)·L1(rgb) + λs·(1-SSIM) + λd·L1(depth).  densification은 gsplat DefaultStrategy.

※ gsplat API(DefaultStrategy/rasterization)는 설치본으로 검증 후 확정.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from pytorch_msssim import ssim as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.backproject import read_colmap_images, read_colmap_cameras, read_points3d  # noqa: E402

C0 = 0.28209479177387814  # SH DC 계수


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / C0


def load_views(proc, device):
    sparse = proc / "colmap" / "sparse" / "0"
    W, H, fx, fy, cx, cy = read_colmap_cameras(sparse / "cameras.txt")
    imgs = read_colmap_images(sparse / "images.txt")
    ds = json.loads((proc / "intrinsics.json").read_text())["depth_scale"]
    assoc = {Path(p[1]).name: p[3] for p in (l.split() for l in open(proc / "associations.txt")) if len(p) >= 4}
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    views = []
    for name, (qw, qx, qy, qz, tx, ty, tz) in imgs.items():
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()   # Tcw (world→cam)
        vm[:3, 3] = [tx, ty, tz]
        rgb = cv2.imread(str(proc / "rgb" / name))[:, :, ::-1].astype(np.float32) / 255.0
        drel = assoc.get(name)
        depth = (cv2.imread(str(proc / drel), cv2.IMREAD_UNCHANGED).astype(np.float32) / ds
                 if drel else np.zeros((H, W), np.float32))
        views.append(dict(name=name,
                          viewmat=torch.tensor(vm, device=device),
                          rgb=torch.tensor(rgb.copy(), device=device),
                          depth=torch.tensor(depth, device=device)))
    return views, K, W, H


def init_gaussians(proc, device, sh_degree):
    xyz, rgb = read_points3d(proc / "colmap" / "sparse" / "0" / "points3D.txt")
    xyz_t = torch.tensor(xyz, device=device)
    rgb_t = torch.tensor(rgb, device=device, dtype=torch.float32) / 255.0
    N = len(xyz)
    knn = cKDTree(xyz).query(xyz, k=4)[0][:, 1:].mean(1)
    scales = torch.log(torch.tensor(knn, device=device, dtype=torch.float32).clamp_min(1e-6))[:, None].repeat(1, 3)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((N,), 0.1, device=device))
    sh0 = rgb_to_sh0(rgb_t)[:, None, :]
    shN = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3, device=device)
    params = torch.nn.ParameterDict(dict(
        means=torch.nn.Parameter(xyz_t), scales=torch.nn.Parameter(scales),
        quats=torch.nn.Parameter(quats), opacities=torch.nn.Parameter(opacities),
        sh0=torch.nn.Parameter(sh0), shN=torch.nn.Parameter(shN),
    )).to(device)
    return params, float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max())


def export_ply(path, params, sh_degree):
    """표준 3DGS PLY (INRIA 포맷: xyz, normals=0, f_dc, f_rest, opacity, scale, rot)."""
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    N = p["means"].shape[0]
    f_dc = p["sh0"].reshape(N, -1)            # [N,3]
    f_rest = p["shN"].reshape(N, -1)          # [N, 3*((sh_degree+1)^2-1)]
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    fields += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([p["means"], np.zeros((N, 3), np.float32), f_dc, f_rest,
                           p["opacities"].reshape(N, 1), p["scales"], p["quats"]], axis=1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for fld in fields:
            f.write(f"property float {fld}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--depth-lambda", type=float, default=0.2)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--holdout-every", type=int, default=0, help="매 N번째 뷰를 학습에서 제외(평가용). 0=없음")
    ap.add_argument("--tag", default="", help="출력 서브디렉토리 접미사 (예: m1)")
    ap.add_argument("--refine-stop", type=int, default=15000, help="densification 중단 step (hold-out 발산 방지로 낮춤)")
    ap.add_argument("--strategy", choices=["default", "mcmc"], default="default",
                    help="densification 전략. mcmc는 cap_max로 가우시안 수 하드캡(천장 안 품질↑)")
    ap.add_argument("--cap-max", type=int, default=2_000_000, help="mcmc 전용: 최대 가우시안 수")
    ap.add_argument("--opacity-reg", type=float, default=0.01, help="mcmc 전용: opacity L1 정규화")
    ap.add_argument("--scale-reg", type=float, default=0.01, help="mcmc 전용: scale L1 정규화")
    ap.add_argument("--exclude-list", type=Path, help="학습/holdout에서 제외할 이미지명 목록(동적/사람 구간 KF)")
    args = ap.parse_args()
    train_loop(args)


def train_loop(args):
    """학습 루프 본체. main()이 인자 파싱 후 호출하거나 orchestrator가 직접 호출."""
    device = "cuda"
    proc = args.root / "data" / "processed" / args.scene
    out = args.root / "outputs" / args.scene / ("gsplat" + (f"_{args.tag}" if args.tag else ""))
    (out / "renders").mkdir(parents=True, exist_ok=True)

    views, K, W, H = load_views(proc, device)
    views.sort(key=lambda v: v["name"])
    if args.exclude_list:
        excl = {l.strip() for l in open(args.exclude_list) if l.strip()}
        n0 = len(views)
        views = [v for v in views if v["name"] not in excl]
        print(f"[train] exclude-list 적용: {n0}→{len(views)} views (제외 {n0 - len(views)})")
    if args.holdout_every > 0:
        holdout = views[:: args.holdout_every]                                  # 매 N번째 = 평가용
        train_views = [v for i, v in enumerate(views) if i % args.holdout_every != 0]
        (out / "holdout.txt").write_text("\n".join(v["name"] for v in holdout) + "\n")
    else:
        holdout, train_views = [], views
    params, scene_scale = init_gaussians(proc, device, args.sh_degree)
    print(f"[train] train={len(train_views)} holdout={len(holdout)} {W}x{H} init={params['means'].shape[0]} scale={scene_scale:.2f}")

    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    optimizers = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}  # 포즈 텐서 없음
    print(f"[train] optimizer param groups (포즈 고정 증거): {sorted(optimizers.keys())}")

    if args.strategy == "mcmc":
        strategy = MCMCStrategy(verbose=False, cap_max=args.cap_max)
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state()
        # MCMC noise는 means LR로 스케일됨(scaler=lr·noise_lr). LR 고정이면 noise가 끝까지 안 식어
        # 그 자체로 soft → 학습 끝에 ~1%까지 지수 감쇠(gsplat MCMC 예제 표준).
        means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.iters))
    else:
        strategy = DefaultStrategy(verbose=False, refine_stop_iter=args.refine_stop)
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)
        means_sched = None
    print(f"[train] strategy={args.strategy}" + (f" cap_max={args.cap_max}" if args.strategy == "mcmc" else ""))

    log = open(out / "train_log.jsonl", "w")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train_views)).tolist()
    for step in range(args.iters):
        v = train_views[order[step % len(train_views)]]
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        sh_deg = min(args.sh_degree, step // (args.iters // (args.sh_degree + 1) + 1))
        render, alphas, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors,
            v["viewmat"][None], K[None], W, H,
            sh_degree=sh_deg, render_mode="RGB+ED", absgrad=False, packed=False)
        rgb_pred = render[0, ..., :3]
        depth_pred = render[0, ..., 3]
        if args.strategy == "default":
            info["means2d"].retain_grad()                                    # MCMC는 2D grad 미사용
            strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = F.l1_loss(rgb_pred, v["rgb"])
        ssim_l = 1 - ssim_fn(rgb_pred.permute(2, 0, 1)[None], v["rgb"].permute(2, 0, 1)[None], data_range=1.0)
        dmask = v["depth"] > 0
        depth_l = F.l1_loss(depth_pred[dmask], v["depth"][dmask]) if dmask.any() else torch.zeros((), device=device)
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * ssim_l + args.depth_lambda * depth_l
        if args.strategy == "mcmc":                                          # MCMC 필수 정규화(없으면 품질 저하)
            loss = loss + args.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean() \
                        + args.scale_reg * torch.exp(params["scales"]).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params["means"]], max_norm=10.0)  # 위치 overshoot 방지(hold-out 무감독 영역 drift)
        for o in optimizers.values():
            o.step(); o.zero_grad(set_to_none=True)
        if args.strategy == "mcmc":
            strategy.step_post_backward(params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"])
            means_sched.step()
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 100 == 0 or step == args.iters - 1:
            psnr = -10 * math.log10(max(F.mse_loss(rgb_pred, v["rgb"]).item(), 1e-10))
            rec = dict(step=step, loss=round(loss.item(), 5), l1=round(l1.item(), 5),
                       depth_l1=round(depth_l.item(), 5), train_psnr=round(psnr, 2),
                       n_gauss=int(params["means"].shape[0]))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{step}] loss={rec['loss']} depth_l1={rec['depth_l1']} psnr={rec['train_psnr']} N={rec['n_gauss']}")
        if step % 5000 == 0 or step == args.iters - 1:
            img = (rgb_pred.clamp(0, 1).detach().cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(str(out / "renders" / f"step{step:06d}.png"), img)
    export_ply(out / "scene.ply", params, args.sh_degree)
    log.close()
    print(f"[train] DONE → {out/'scene.ply'} ({params['means'].shape[0]} gaussians)")


if __name__ == "__main__":
    main()
