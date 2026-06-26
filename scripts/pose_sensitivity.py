#!/usr/bin/env python3
"""pose_sensitivity.py — 합쳐진 맵의 '포즈 허용오차' 곡선 (thesis 핵심 숫자).

가우시안 맵의 holdout 포즈를 알고 있는 양만큼 틀어(localizer 오차 모사) → 그 포즈로 렌더 →
실제 프레임과 PSNR/SSIM/LPIPS 비교. 결과 곡선 "렌더 화질 vs 로컬라이제이션 오차" =
로컬라이저가 맞춰야 할 정확도 예산 + photometric reloc(B)의 수렴 basin.

home은 포즈 GT가 없으므로 기준 포즈 = COLMAP 포즈, 지표 = render-vs-real(이미지가 정답).
usage: python pose_sensitivity.py [--tag mcmc2m] [--frames 12] [--dirs 4] [--scale 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as Rot
import lpips as lpips_lib
from pytorch_msssim import ssim as ssim_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.gsplat_io import load_ply, render  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402

DEV = "cuda"
_LP = None


def lpips_metric(pred, gt):
    global _LP
    if _LP is None:
        _LP = lpips_lib.LPIPS(net="alex", verbose=False).to(DEV).eval()
    with torch.no_grad():
        return float(_LP(pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item())


def metrics(pred, gt):
    psnr = float(-10 * np.log10(max(((pred - gt) ** 2).mean().item(), 1e-10)))
    ss = float(ssim_fn(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0))
    return psnr, ss, lpips_metric(pred, gt)


def perturb_viewmat(vm, trans_m, rot_deg, rng):
    """카메라를 월드에서 trans_m(랜덤 방향) 이동 + rot_deg(랜덤 축) 회전시킨 viewmat 반환."""
    C = np.linalg.inv(vm)                                  # cam→world
    if trans_m > 0:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        C[:3, 3] += d * trans_m
    if rot_deg > 0:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        dR = Rot.from_rotvec(ax * np.deg2rad(rot_deg)).as_matrix()
        C[:3, :3] = dR @ C[:3, :3]
    return np.linalg.inv(C).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--tag", default="mcmc2m")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--dirs", type=int, default=4, help="각 크기마다 랜덤 방향 N개 평균")
    ap.add_argument("--scale", type=float, default=0.5, help="렌더 해상도 배율(속도)")
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
    print(f"[sens] {len(holdout)} frames, {args.dirs} dirs/mag, {W}x{H}, asset={args.tag}", flush=True)

    TRANS = [0.0, 0.01, 0.02, 0.05, 0.10]   # m
    ROT = [0.0, 1.0, 2.0, 5.0]              # deg
    real = {}
    for n in holdout:
        gt = cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0
        real[n] = torch.tensor(cv2.resize(gt, (W, H)).copy(), device=DEV)

    def sweep(kind, mags):
        rows = []
        for m in mags:
            acc = []
            for n in holdout:
                qw, qx, qy, qz, tx, ty, tz = imgs[n]
                vm = np.eye(4, dtype=np.float32)
                vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix(); vm[:3, 3] = [tx, ty, tz]
                ndir = 1 if m == 0 else args.dirs
                for _ in range(ndir):
                    tm = m if kind == "trans" else 0.0
                    rd = m if kind == "rot" else 0.0
                    vmp = perturb_viewmat(vm, tm, rd, rng)
                    pred = render(g, torch.tensor(vmp, device=DEV), K, W, H).clamp(0, 1)
                    acc.append(metrics(pred, real[n]))
            a = np.array(acc)
            rows.append(dict(mag=m, psnr=round(float(np.median(a[:, 0])), 2),
                             ssim=round(float(np.median(a[:, 1])), 4),
                             lpips=round(float(np.median(a[:, 2])), 4)))
            print(f"  [{kind}] {m}: PSNR {rows[-1]['psnr']} SSIM {rows[-1]['ssim']} LPIPS {rows[-1]['lpips']}", flush=True)
        return rows

    out = dict(asset=args.tag, frames=len(holdout), dirs=args.dirs,
               trans_cm=sweep("trans", TRANS), rot_deg=sweep("rot", ROT))
    (gdir / "pose_sensitivity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # 곡선 플롯(matplotlib 있으면)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        t = out["trans_cm"]; r = out["rot_deg"]
        ax[0].plot([x["mag"] * 100 for x in t], [x["psnr"] for x in t], "o-", label="PSNR")
        ax[0].set_xlabel("translation error (cm)"); ax[0].set_ylabel("PSNR (dB)"); ax[0].set_title("포즈 이동오차 vs 렌더"); ax[0].grid(True)
        ax2 = ax[0].twinx(); ax2.plot([x["mag"] * 100 for x in t], [x["lpips"] for x in t], "s--", color="r", label="LPIPS"); ax2.set_ylabel("LPIPS", color="r")
        ax[1].plot([x["mag"] for x in r], [x["psnr"] for x in r], "o-")
        ax[1].set_xlabel("rotation error (deg)"); ax[1].set_ylabel("PSNR (dB)"); ax[1].set_title("포즈 회전오차 vs 렌더"); ax[1].grid(True)
        ax3 = ax[1].twinx(); ax3.plot([x["mag"] for x in r], [x["lpips"] for x in r], "s--", color="r"); ax3.set_ylabel("LPIPS", color="r")
        fig.tight_layout()
        op = ROOT / "docs/assets/report/pose_sensitivity_curve.png"; fig.savefig(op, dpi=110)
        print(f"[sens] curve → {op}", flush=True)
    except Exception as e:
        print(f"[sens] plot skip: {e}", flush=True)
    print("[sens] DONE", flush=True)


if __name__ == "__main__":
    main()
