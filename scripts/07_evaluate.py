#!/usr/bin/env python3
"""07_evaluate.py — Phase 6 평가 + M1 베이스라인 비교 (ORB-SLAM3 포즈 vs COLMAP SfM 포즈).

- hold-out(매 8번째, 학습 제외) PSNR/SSIM/LPIPS — 양쪽 모델, **per-view 분포**. 공통 hold-out·동일 프로토콜.
- **ATE-vs-GT**: 양쪽 포즈를 TUM groundtruth.txt에 Sim3 정렬 후 RMSE — COLMAP이 신뢰할 baseline인지 입증.
- **M1 판정**: |median(PSNR_ORB) − median(PSNR_COLMAP)| ≤ 0.5dB **AND** 양쪽 ATE 합리적(COLMAP이 strawman 아님).

usage: 07_evaluate.py [--tag m1] [--gt <groundtruth.txt>]
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as Rot
import lpips as lpips_lib
from pytorch_msssim import ssim as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, render  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from pipeline.colmap_convert import load_tum, camera_center  # noqa: E402

DEV = "cuda"
_LP = None


def lpips_metric(pred, gt_np):
    global _LP
    if _LP is None:
        _LP = lpips_lib.LPIPS(net="alex", verbose=False).to(DEV).eval()
    p = pred.permute(2, 0, 1)[None] * 2 - 1
    g = torch.tensor(gt_np.copy(), device=DEV).permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        return float(_LP(p, g).item())


def eval_holdout(scene, tag, root):
    proc = root / "data" / "processed" / scene
    gdir = root / "outputs" / scene / f"gsplat_{tag}"
    g = load_ply(gdir / "scene.ply", DEV)
    sp = proc / "colmap" / "sparse" / "0"
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    imgs = read_colmap_images(sp / "images.txt")
    holdout = [l.strip() for l in open(gdir / "holdout.txt") if l.strip()]
    res = []
    for name in holdout:
        if name not in imgs:
            continue
        qw, qx, qy, qz, tx, ty, tz = imgs[name]
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        vm[:3, 3] = [tx, ty, tz]
        pred = render(g, torch.tensor(vm, device=DEV), K, W, H)
        gt = cv2.imread(str(proc / "rgb" / name))[:, :, ::-1].astype(np.float32) / 255.0
        psnr = float(-10 * np.log10(max(((pred.detach().cpu().numpy() - gt) ** 2).mean(), 1e-10)))
        ss = float(ssim_fn(pred.permute(2, 0, 1)[None],
                           torch.tensor(gt.copy(), device=DEV).permute(2, 0, 1)[None], data_range=1.0))
        res.append(dict(name=name, psnr=round(psnr, 2), ssim=round(ss, 4), lpips=round(lpips_metric(pred, gt), 4)))
    return res, holdout


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Sc.T @ Dc / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / ((Sc ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def ate_rmse(est, gt):
    s, R, t = umeyama(est, gt)
    aligned = (s * (R @ est.T).T + t)
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))


def gt_centers(ts_list, gt_path):
    G = [(float(l.split()[0]), np.array([float(x) for x in l.split()[1:4]]))
         for l in open(gt_path) if l.strip() and not l.startswith("#")]
    gts = [x[0] for x in G]
    out = []
    for ts in ts_list:
        i = bisect.bisect_left(gts, ts)
        j = min([k for k in (i - 1, i) if 0 <= k < len(gts)], key=lambda k: abs(gts[k] - ts))
        out.append(G[j][1])
    return np.array(out)


def orb_centers_ts(root, scene):
    kfs = load_tum(root / "outputs" / scene / "slam" / "KeyFrameTrajectory.txt")
    c = np.array([[k[1], k[2], k[3]] for k in kfs]); ts = [k[0] for k in kfs]
    return c, ts


def colmap_centers_ts(root, scene):
    imgs = read_colmap_images(root / "data" / "processed" / scene / "colmap" / "sparse" / "0" / "images.txt")
    c, ts = [], []
    for name, p in imgs.items():
        c.append(camera_center(*p)); ts.append(float(Path(name).stem))
    return np.array(c), ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--orb-scene", default="tum_fr1_desk")
    ap.add_argument("--colmap-scene", default="tum_fr1_desk_sfm")
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--gt", type=Path,
                    default=Path("/home/ingon/docker_share/rtabmap/demo/rgbd_dataset_freiburg1_desk/groundtruth.txt"))
    args = ap.parse_args()

    orb_res, orb_ho = eval_holdout(args.orb_scene, args.tag, args.root)
    col_res, col_ho = eval_holdout(args.colmap_scene, args.tag, args.root)
    common = sorted(set(r["name"] for r in orb_res) & set(r["name"] for r in col_res))  # 교집합 hold-out
    om = {r["name"]: r for r in orb_res}; cm = {r["name"]: r for r in col_res}
    op = np.array([om[n]["psnr"] for n in common]); cp = np.array([cm[n]["psnr"] for n in common])
    delta = abs(float(np.median(op)) - float(np.median(cp)))

    # ATE
    oc, ots = orb_centers_ts(args.root, args.orb_scene)
    cc, cts = colmap_centers_ts(args.root, args.colmap_scene)
    orb_ate = ate_rmse(oc, gt_centers(ots, args.gt))
    col_ate = ate_rmse(cc, gt_centers(cts, args.gt))

    def summ(p, m):
        return dict(psnr_median=round(float(np.median(p)), 2), psnr_mean=round(float(np.mean(p)), 2),
                    psnr_min=round(float(np.min(p)), 2),
                    ssim_median=round(float(np.median([m[n]["ssim"] for n in common])), 4),
                    lpips_median=round(float(np.median([m[n]["lpips"] for n in common])), 4))
    out = dict(
        holdout_n=len(common),
        orb=summ(op, om), colmap=summ(cp, cm),
        psnr_delta=round(delta, 3),
        ate_orb_m=round(orb_ate, 4), ate_colmap_m=round(col_ate, 4),
        per_view={n: dict(orb=om[n]["psnr"], colmap=cm[n]["psnr"]) for n in common},
        verdict_psnr=bool(delta <= 0.5),
        # COLMAP이 신뢰할 baseline인가: ATE가 합리적(<10cm)이고 ORB와 동급(±2x 이내)
        verdict_colmap_credible=bool(col_ate < 0.10 and col_ate < 2 * orb_ate + 0.01),
    )
    out["M1_PASS"] = bool(out["verdict_psnr"] and out["verdict_colmap_credible"])
    odir = args.root / "outputs" / args.orb_scene
    (odir / "eval_m1.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n===== M1 평가 (공통 hold-out {len(common)}뷰) =====")
    print(f"  ORB    PSNR median={out['orb']['psnr_median']} mean={out['orb']['psnr_mean']} min={out['orb']['psnr_min']} | SSIM {out['orb']['ssim_median']} LPIPS {out['orb']['lpips_median']}")
    print(f"  COLMAP PSNR median={out['colmap']['psnr_median']} mean={out['colmap']['psnr_mean']} min={out['colmap']['psnr_min']} | SSIM {out['colmap']['ssim_median']} LPIPS {out['colmap']['lpips_median']}")
    print(f"  PSNR Δ(median) = {out['psnr_delta']} dB  → {'≤0.5 ✅' if out['verdict_psnr'] else '>0.5 ❌'}")
    print(f"  ATE-vs-GT: ORB={out['ate_orb_m']*100:.1f}cm  COLMAP={out['ate_colmap_m']*100:.1f}cm  → COLMAP 신뢰 baseline? {'✅' if out['verdict_colmap_credible'] else '⚠'}")
    print(f"\n=> M1 {'PASS ✅' if out['M1_PASS'] else 'FAIL/보류'}  (PSNR-Δ≤0.5dB AND COLMAP 신뢰)")
    sys.exit(0 if out["M1_PASS"] else 1)


if __name__ == "__main__":
    main()
