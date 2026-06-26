#!/usr/bin/env python3
"""basin_map.py — photometric reloc 수렴 basin 매핑.

각 초기 포즈 오차 설정에서 photometric_reloc의 핵심 함수를 재사용해
수렴 여부를 집계하고, 이동/회전 단독 basin 곡선을 그린다.

usage: python basin_map.py [--frames 8] [--iters 200] [--scale 0.5] [--tag mcmc2m]
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.gsplat_io import load_ply  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
# photometric_reloc의 핵심 함수 재사용 (scene.ply 중복 로딩 방지)
from scripts.photometric_reloc import se3_exp, render_at, pose_err, relocalize  # noqa: E402

DEV = "cuda"

# 수렴 판정 임계 (이봉형 분포 골 근처)
CONV_TRANS_CM = 1.0   # 최종 이동 오차 < 1 cm
CONV_ROT_DEG  = 1.0   # 최종 회전 오차 < 1°

# sweep 설정: (init_trans_m, init_rot_deg, label)
CONFIGS = [
    # 이동 단독
    (0.05, 0.0, "T05"),
    (0.10, 0.0, "T10"),
    (0.20, 0.0, "T20"),
    (0.40, 0.0, "T40"),
    # 회전 단독
    (0.0,  3.0, "R03"),
    (0.0,  6.0, "R06"),
    (0.0, 12.0, "R12"),
    (0.0, 24.0, "R24"),
    # 대각 복합
    (0.05,  3.0, "D05_03"),
    (0.10,  6.0, "D10_06"),
    (0.20, 12.0, "D20_12"),
]


def run_config(g, holdout, proc, K, W, H, args, init_trans_m, init_rot_deg):
    """한 설정에 대해 모든 프레임 수행 → 요약 dict 반환."""
    imgs = read_colmap_images(proc / "colmap/sparse/0/images.txt")
    # config마다 동일 seed → 동일 perturbation 방향·축 (크기만 달라짐)
    rng = np.random.default_rng(0)
    res = []
    for n in holdout:
        qw, qx, qy, qz, tx, ty, tz = imgs[n]
        vm_ref = np.eye(4, dtype=np.float32)
        vm_ref[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        vm_ref[:3, 3] = [tx, ty, tz]

        # 초기 오차 모사
        C = np.linalg.inv(vm_ref)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        C[:3, 3] += d * init_trans_m
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        if init_rot_deg > 0:
            C[:3, :3] = Rot.from_rotvec(ax * np.deg2rad(init_rot_deg)).as_matrix() @ C[:3, :3]
        vm_init = np.linalg.inv(C).astype(np.float32)

        gt = cv2.resize(
            cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0,
            (W, H))
        gt_t = torch.tensor(gt.copy(), device=DEV)
        psnr0 = float(-10 * np.log10(max(F.mse_loss(
            render_at(g, torch.tensor(vm_init, device=DEV), K, W, H).clamp(0, 1), gt_t).item(), 1e-10)))
        dt0, dr0 = pose_err(vm_init, vm_ref)

        vm_fin, psnr1 = relocalize(g, vm_init, gt, K, W, H, args.iters)
        dt1, dr1 = pose_err(vm_fin, vm_ref)
        res.append(dict(name=n,
                        init_trans_cm=round(dt0, 2), init_rot_deg=round(dr0, 2),
                        final_trans_cm=round(dt1, 2), final_rot_deg=round(dr1, 2),
                        psnr_init=round(psnr0, 2), psnr_final=round(psnr1, 2)))
        print(f"    {n}: {dt0:.1f}cm/{dr0:.1f}° -> {dt1:.1f}cm/{dr1:.1f}°  "
              f"PSNR {psnr0:.1f}->{psnr1:.1f}", flush=True)

    a = lambda k: np.array([r[k] for r in res])
    conv = (a("final_trans_cm") < CONV_TRANS_CM) & (a("final_rot_deg") < CONV_ROT_DEG)
    return dict(
        init_trans_m=init_trans_m,
        init_rot_deg=init_rot_deg,
        nominal_trans_cm=round(init_trans_m * 100, 1),
        nominal_rot_deg=round(init_rot_deg, 1),
        converged_frac=round(float(conv.mean()), 3),
        pose_trans_cm=dict(
            init=round(float(np.median(a("init_trans_cm"))), 2),
            final=round(float(np.median(a("final_trans_cm"))), 2)),
        pose_rot_deg=dict(
            init=round(float(np.median(a("init_rot_deg"))), 2),
            final=round(float(np.median(a("final_rot_deg"))), 2)),
        psnr=dict(
            init=round(float(np.median(a("psnr_init"))), 2),
            final=round(float(np.median(a("psnr_final"))), 2)),
        per_frame=res)


def plot_basin(results_by_label, out_path):
    trans_pts = [(r["nominal_trans_cm"], r["converged_frac"])
                 for lbl, r in results_by_label.items() if r["init_rot_deg"] == 0.0]
    rot_pts   = [(r["nominal_rot_deg"],  r["converged_frac"])
                 for lbl, r in results_by_label.items() if r["init_trans_m"] == 0.0]
    trans_pts.sort(); rot_pts.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if trans_pts:
        xs, ys = zip(*trans_pts)
        ax1.plot(xs, ys, "o-", color="steelblue")
        ax1.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90%")
        ax1.set_xlabel("Initial translation error (cm)")
        ax1.set_ylabel("Converged fraction (<1cm & <1deg)")
        ax1.set_title("Translation-only sweep")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()

    if rot_pts:
        xs, ys = zip(*rot_pts)
        ax2.plot(xs, ys, "o-", color="darkorange")
        ax2.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90%")
        ax2.set_xlabel("Initial rotation error (deg)")
        ax2.set_ylabel("Converged fraction (<1cm & <1deg)")
        ax2.set_title("Rotation-only sweep")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()

    fig.suptitle("Photometric Reloc Convergence Basin (200 iters)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[basin] plot saved: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--tag", default="mcmc2m")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    proc = ROOT / "data/processed" / args.scene
    gdir = ROOT / "outputs" / args.scene / f"gsplat_{args.tag}"
    sp   = proc / "colmap/sparse/0"

    W0, H0, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    s = args.scale
    W, H = int(W0 * s), int(H0 * s)
    K = torch.tensor([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]], device=DEV)

    imgs = read_colmap_images(sp / "images.txt")
    # scene.ply 한 번만 로딩
    g = load_ply(gdir / "scene.ply", DEV)

    holdout = [l.strip() for l in open(gdir / "holdout.txt") if l.strip() and l.strip() in imgs]
    holdout = holdout[:: max(1, len(holdout) // args.frames)][:args.frames]
    print(f"[basin] {len(holdout)} frames, {args.iters} iters, {W}x{H}, {len(CONFIGS)} configs", flush=True)

    results_by_label = {}
    for i, (init_trans_m, init_rot_deg, label) in enumerate(CONFIGS):
        print(f"\n[basin] ({i+1}/{len(CONFIGS)}) {label}: "
              f"trans={init_trans_m*100:.0f}cm  rot={init_rot_deg:.0f}deg", flush=True)
        r = run_config(g, holdout, proc, K, W, H, args, init_trans_m, init_rot_deg)
        results_by_label[label] = r
        print(f"[basin] {label}: converged_frac={r['converged_frac']:.2f}  "
              f"final_median {r['pose_trans_cm']['final']}cm/{r['pose_rot_deg']['final']}deg  "
              f"PSNR {r['psnr']['final']}", flush=True)

    # 결과 JSON 저장
    out_json = gdir / "convergence_basin.json"
    payload = dict(
        frames=len(holdout), iters=args.iters, scale=args.scale,
        conv_thresh=dict(trans_cm=CONV_TRANS_CM, rot_deg=CONV_ROT_DEG),
        results=results_by_label)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[basin] json saved: {out_json}", flush=True)

    # 곡선 플롯
    plot_dir = ROOT / "docs/assets/report"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_basin(results_by_label, plot_dir / "convergence_basin.png")

    # 요약 출력
    print("\n=== Basin sweep 결과 ===")
    header = f"{'Label':10s}  {'Trans(cm)':>10s}  {'Rot(deg)':>8s}  {'Conv%':>6s}  "
    header += f"{'FinalT(cm)':>10s}  {'FinalR(deg)':>11s}  {'PSNR':>6s}"
    print(header)
    for lbl, r in results_by_label.items():
        print(f"{lbl:10s}  {r['nominal_trans_cm']:>10.1f}  {r['nominal_rot_deg']:>8.1f}  "
              f"{r['converged_frac']*100:>5.0f}%  "
              f"{r['pose_trans_cm']['final']:>10.2f}  {r['pose_rot_deg']['final']:>11.2f}  "
              f"{r['psnr']['final']:>6.1f}")
    print("[basin] DONE", flush=True)


if __name__ == "__main__":
    main()
