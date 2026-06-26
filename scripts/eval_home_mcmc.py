#!/usr/bin/env python3
"""eval_home_mcmc.py — home 자산 품질 A/B (default vs MCMC cap 스윕).

같은 scene·같은 holdout 프로토콜로 여러 gsplat 디렉토리를 PSNR/SSIM/LPIPS 비교.
"soft"는 공간 선명도 문제이고 PSNR은 blur에 둔감 → **SSIM↑·LPIPS↓**가 1차 판정 지표.
(07_evaluate.py의 main은 M1 전용 2-scene ATE 비교라 재사용 불가 → 메트릭만 차용.)

usage: eval_home_mcmc.py [--scene S] [--dirs gsplat gsplat_mcmc2m gsplat_mcmc3m]
"""
import argparse
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


def eval_dir(proc, gdir, imgs, K, W, H, holdout):
    g = load_ply(gdir / "scene.ply", DEV)
    n_gauss = g["means"].shape[0]
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
        res.append(dict(name=name, psnr=psnr, ssim=ss, lpips=lpips_metric(pred, gt)))
    return res, n_gauss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--dirs", nargs="+", default=["gsplat", "gsplat_mcmc2m", "gsplat_mcmc3m"])
    args = ap.parse_args()
    proc = args.root / "data" / "processed" / args.scene
    sp = proc / "colmap" / "sparse" / "0"
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    imgs = read_colmap_images(sp / "images.txt")

    # 공통 holdout = 존재하는 디렉토리들의 holdout.txt 교집합(동일 프로토콜이라 통상 동일)
    avail = [(d, args.root / "outputs" / args.scene / d) for d in args.dirs
             if (args.root / "outputs" / args.scene / d / "scene.ply").exists()]
    if not avail:
        sys.exit(f"평가할 자산 없음: {args.dirs}")
    holdsets = [set(l.strip() for l in open(gd / "holdout.txt") if l.strip())
                for _, gd in avail if (gd / "holdout.txt").exists()]
    holdout = sorted(set.intersection(*holdsets)) if holdsets else []

    print(f"\n===== home 자산 A/B (scene={args.scene}, 공통 holdout {len(holdout)}뷰) =====")
    print(f"{'asset':>16} | {'N_gauss':>9} | {'PSNR med':>8} | {'SSIM med':>8} | {'LPIPS med':>9}  (SSIM↑·LPIPS↓ = 선명)")
    rows = []
    for label, gd in avail:
        res, ng = eval_dir(proc, gd, imgs, K, W, H, holdout)
        if not res:
            continue
        p = np.median([r["psnr"] for r in res]); s = np.median([r["ssim"] for r in res])
        lp = np.median([r["lpips"] for r in res])
        rows.append((label, ng, p, s, lp))
        print(f"{label:>16} | {ng:>9,} | {p:>8.2f} | {s:>8.4f} | {lp:>9.4f}")

    if len(rows) >= 2:
        base = rows[0]
        print(f"\n기준={base[0]} 대비 Δ(선명도):")
        for label, ng, p, s, lp in rows[1:]:
            tag = "선명↑" if (s > base[3] and lp < base[4]) else ("모호" if (s > base[3]) != (lp < base[4]) else "흐림↓")
            print(f"  {label}: ΔPSNR {p-base[2]:+.2f}  ΔSSIM {s-base[3]:+.4f}  ΔLPIPS {lp-base[4]:+.4f}  → {tag}")


if __name__ == "__main__":
    main()
