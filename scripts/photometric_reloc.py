#!/usr/bin/env python3
"""photometric_reloc.py — 가우시안 맵 자체로 위치추정 (B: 통합 프론티어).

'합쳐진 맵'의 핵심 증명: 가우시안 맵을 미분가능 렌더러로 두고, 틀어진 초기 포즈에서 시작해
렌더 이미지가 실제 쿼리와 일치하도록 6DoF 카메라 포즈를 최적화(iNeRF/analysis-by-synthesis).
포즈 오차가 줄고 render-vs-real이 회복되면 → 가우시안 맵이 *로컬라이저 역할*을 함 = 하나의 맵이 둘 다.

home은 포즈 GT 없음 → 기준 포즈=COLMAP 포즈를 known Δ만큼 틀어 '초기 localizer 오차'를 모사,
photometric 최적화가 그 Δ를 얼마나 회복하는지(포즈오차 ↓) + render-vs-real PSNR ↑를 측정.
지표의 정답은 실제 이미지(GT 포즈 불요). usage 아래.

usage: python photometric_reloc.py [--tag mcmc2m] [--frames 10] [--init-trans 0.05 --init-rot 3]
                                    [--iters 300] [--scale 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.gsplat_io import load_ply  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from gsplat import rasterization  # noqa: E402

DEV = "cuda"


def se3_exp(xi):
    """xi[6]=(rho, phi) → SE(3) 4x4 (정확 exp). 포즈 delta 파라미터화."""
    rho, phi = xi[:3], xi[3:]
    th = torch.linalg.norm(phi) + 1e-12
    Kx = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    Kx[0, 1], Kx[0, 2] = -phi[2], phi[1]
    Kx[1, 0], Kx[1, 2] = phi[2], -phi[0]
    Kx[2, 0], Kx[2, 1] = -phi[1], phi[0]
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    R = I + torch.sin(th) / th * Kx + (1 - torch.cos(th)) / th**2 * (Kx @ Kx)
    V = I + (1 - torch.cos(th)) / th**2 * Kx + (th - torch.sin(th)) / th**3 * (Kx @ Kx)
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def render_at(g, viewmat, K, W, H):
    out, _, _ = rasterization(
        g["means"], g["quats"], torch.exp(g["scales"]), torch.sigmoid(g["opacities"]),
        g["sh"], viewmat[None], K[None], W, H, sh_degree=g["sh_degree"], render_mode="RGB")
    return out[0, ..., :3]


def pose_err(vm_a, vm_b):
    """두 viewmat 사이 카메라 중심거리(cm)·회전각(deg)."""
    Ca, Cb = np.linalg.inv(vm_a), np.linalg.inv(vm_b)
    dt = np.linalg.norm(Ca[:3, 3] - Cb[:3, 3]) * 100
    dr = np.rad2deg(np.linalg.norm(Rot.from_matrix(Cb[:3, :3] @ Ca[:3, :3].T).as_rotvec()))
    return dt, dr


def relocalize(g, vm_init, gt_img, K, W, H, iters):
    """vm_init에서 시작, render가 gt_img와 맞도록 se3 delta 최적화. 최종 viewmat 반환."""
    base = torch.tensor(vm_init, device=DEV)
    xi = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=3e-3)
    gt = torch.tensor(gt_img.copy(), device=DEV)
    for _ in range(iters):
        vm = se3_exp(xi) @ base
        pred = render_at(g, vm, K, W, H).clamp(0, 1)
        loss = F.l1_loss(pred, gt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        vm_final = (se3_exp(xi) @ base).detach().cpu().numpy()
        final_psnr = float(-10 * np.log10(max(F.mse_loss(
            render_at(g, torch.tensor(vm_final, device=DEV), K, W, H).clamp(0, 1), gt).item(), 1e-10)))
    return vm_final, final_psnr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--tag", default="mcmc2m")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--init-trans", type=float, default=0.05, help="초기 포즈 오차(m) 모사")
    ap.add_argument("--init-rot", type=float, default=3.0, help="초기 포즈 오차(deg) 모사")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    proc = ROOT / "data/processed" / args.scene
    gdir = ROOT / "outputs" / args.scene / f"gsplat_{args.tag}"
    sp = proc / "colmap/sparse/0"
    W0, H0, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    s = args.scale
    W, H = int(W0 * s), int(H0 * s)
    K = torch.tensor([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]], device=DEV)
    imgs = read_colmap_images(sp / "images.txt")
    g = load_ply(gdir / "scene.ply", DEV)
    holdout = [l.strip() for l in open(gdir / "holdout.txt") if l.strip() and l.strip() in imgs]
    holdout = holdout[:: max(1, len(holdout) // args.frames)][:args.frames]
    print(f"[reloc] {len(holdout)} frames, init Δ≈{args.init_trans*100:.0f}cm/{args.init_rot:.0f}°, "
          f"{args.iters} iters, {W}x{H}, asset={args.tag}", flush=True)

    res = []
    for n in holdout:
        qw, qx, qy, qz, tx, ty, tz = imgs[n]
        vm_ref = np.eye(4, dtype=np.float32)
        vm_ref[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix(); vm_ref[:3, 3] = [tx, ty, tz]
        # 초기 오차 모사: 카메라 중심 이동 + 회전
        C = np.linalg.inv(vm_ref)
        d = rng.normal(size=3); d /= np.linalg.norm(d); C[:3, 3] += d * args.init_trans
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        C[:3, :3] = Rot.from_rotvec(ax * np.deg2rad(args.init_rot)).as_matrix() @ C[:3, :3]
        vm_init = np.linalg.inv(C).astype(np.float32)

        gt = cv2.resize(cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0, (W, H))
        gt_t = torch.tensor(gt.copy(), device=DEV)
        psnr0 = float(-10 * np.log10(max(F.mse_loss(
            render_at(g, torch.tensor(vm_init, device=DEV), K, W, H).clamp(0, 1), gt_t).item(), 1e-10)))
        dt0, dr0 = pose_err(vm_init, vm_ref)

        vm_fin, psnr1 = relocalize(g, vm_init, gt, K, W, H, args.iters)
        dt1, dr1 = pose_err(vm_fin, vm_ref)
        res.append(dict(name=n, init_trans_cm=round(dt0, 2), init_rot_deg=round(dr0, 2),
                        final_trans_cm=round(dt1, 2), final_rot_deg=round(dr1, 2),
                        psnr_init=round(psnr0, 2), psnr_final=round(psnr1, 2)))
        print(f"  {n}: pose {dt0:.1f}cm/{dr0:.1f}° → {dt1:.1f}cm/{dr1:.1f}°   "
              f"PSNR {psnr0:.1f} → {psnr1:.1f}", flush=True)

    a = lambda k: np.array([r[k] for r in res])
    summ = dict(
        frames=len(res), init_trans=args.init_trans, init_rot=args.init_rot, iters=args.iters,
        pose_trans_cm=dict(init=round(float(np.median(a("init_trans_cm"))), 2),
                           final=round(float(np.median(a("final_trans_cm"))), 2)),
        pose_rot_deg=dict(init=round(float(np.median(a("init_rot_deg"))), 2),
                          final=round(float(np.median(a("final_rot_deg"))), 2)),
        psnr=dict(init=round(float(np.median(a("psnr_init"))), 2),
                  final=round(float(np.median(a("psnr_final"))), 2)),
        converged_frac=round(float((a("final_trans_cm") < 1.0).mean()), 2),  # <1cm 수렴 비율
        per_frame=res)
    (gdir / "photometric_reloc.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(f"\n[reloc] 요약(median): 포즈 {summ['pose_trans_cm']['init']}cm/{summ['pose_rot_deg']['init']}° "
          f"→ {summ['pose_trans_cm']['final']}cm/{summ['pose_rot_deg']['final']}°  | "
          f"PSNR {summ['psnr']['init']} → {summ['psnr']['final']}  | "
          f"<1cm 수렴 {summ['converged_frac']*100:.0f}%", flush=True)
    print("[reloc] DONE", flush=True)


if __name__ == "__main__":
    main()
