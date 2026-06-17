#!/usr/bin/env python3
"""prune_floaters.py — 학습된 3DGS .ply에서 floater(바늘 잔상 + 방 밖 점) 제거 → SuperSplat용 클린본.

SuperSplat 자유회전은 학습 시점 밖 floater를 다 드러냄. 3가지 필터:
  1) 공간 크롭: 카메라 궤적 bbox + margin 밖 점 제거(멀리 튄 floater).
  2) 스케일 prune: 거대 스케일(바늘) 상위 outlier 제거.
  3) opacity prune: 낮은 불투명도 제거.
usage: prune_floaters.py --scene <s> --model <scene.ply> --out <clean.ply> [--margin 1.0] [--scale-pct 99]
"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, save_ply  # noqa: E402
from pipeline.backproject import read_colmap_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--margin", type=float, default=1.0, help="카메라 bbox 바깥 여유[m]")
    ap.add_argument("--scale-pct", type=float, default=99.0, help="이 백분위 초과 스케일=바늘 제거")
    ap.add_argument("--opacity", type=float, default=0.1, help="sigmoid opacity 임계")
    args = ap.parse_args()

    g = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in load_ply(args.model, "cpu").items()}
    m = g["means"]; N = len(m)
    # 카메라 중심 bbox
    imgs = read_colmap_images(args.root / "data/processed" / args.scene / "colmap/sparse/0/images.txt")
    Cs = np.array([(-Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ np.array(q[4:7])) for q in imgs.values()])
    lo, hi = Cs.min(0) - args.margin, Cs.max(0) + args.margin
    in_box = np.all((m >= lo) & (m <= hi), axis=1)
    # 스케일 outlier(바늘): exp(scale) 최대축이 상위 pct 초과
    smax = np.exp(g["scales"]).max(1)
    small = smax < np.percentile(smax, args.scale_pct)
    # opacity
    opaque = (1 / (1 + np.exp(-g["opacities"].reshape(-1)))) > args.opacity
    keep = in_box & small & opaque
    print(f"[prune] {N} → {keep.sum()} (크롭 {in_box.sum()}, 스케일 {small.sum()}, opacity {opaque.sum()})")

    out = {k: g[k][keep] for k in ("means", "quats", "scales", "opacities", "sh")}
    save_ply(args.out, out)
    print(f"[prune] → {args.out}  ({out['means'].shape[0]} gauss, {args.out.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
