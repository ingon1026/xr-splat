#!/usr/bin/env python3
"""bake_trajectory_ply.py — SLAM 키프레임 궤적+프러스텀을 dense 가우시안으로 만들어 scene.ply에 합침
(Part 2, 출력 B — SuperSplat 인터랙티브). 합본 .ply를 웹뷰어에 드래그하면 가우시안 씬 위에 경로가 보임.

선분을 isotropic 가우시안으로 샘플(간격≤2*radius). gsplat_io.save_ply 컨벤션 준수:
  scales=log(radius), opacities=logit(+9≈불투명), quats=[1,0,0,0], flat SH0 색(f_dc=(rgb-0.5)/C0, f_rest=0).

usage: bake_trajectory_ply.py --scene <s> --model <scene.ply> [--frustum-stride 5] [--line-radius-frac 0.003]
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, save_ply  # noqa: E402
from pipeline.overlay_geometry import build_segments, sample_points  # noqa: E402

C0 = 0.28209479177387814  # SH DC 계수


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--images", type=Path, help="기본: data/processed/<scene>/colmap/sparse/0/images.txt")
    ap.add_argument("--frustum-stride", type=int, default=5)
    ap.add_argument("--frustum-size", type=float, default=1.0)
    ap.add_argument("--line-radius-frac", type=float, default=0.003, help="means bbox 대각 대비 선 가우시안 반경")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sp = args.root / "data" / "processed" / args.scene / "colmap" / "sparse" / "0"
    images = args.images or (sp / "images.txt")
    g = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in load_ply(args.model, "cpu").items()}
    Kc = g["sh"].shape[1]
    segs, d, traj_diag = build_segments(images, sp / "cameras.txt", args.frustum_stride, args.frustum_size)
    radius = args.line_radius_frac * traj_diag             # 궤적 extent 기준(가우시안 bbox는 floater로 부풀려짐)
    pts, cols = sample_points(segs, radius)
    M = len(pts)
    lsh = np.zeros((M, Kc, 3), np.float32); lsh[:, 0, :] = (cols - 0.5) / C0
    line = dict(
        means=pts.astype(np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (M, 1)),
        scales=np.full((M, 3), np.log(radius), np.float32),
        opacities=np.full(M, 9.0, np.float32),
        sh=lsh)
    print(f"[bake] scene N={g['means'].shape[0]}  line pts={M}  radius={radius:.4f}  near_d={d:.3f}  K={Kc}")

    out = dict(
        means=np.concatenate([g["means"], line["means"]], 0),
        quats=np.concatenate([g["quats"], line["quats"]], 0),
        scales=np.concatenate([g["scales"], line["scales"]], 0),
        opacities=np.concatenate([g["opacities"].reshape(-1), line["opacities"]], 0),
        sh=np.concatenate([g["sh"], line["sh"]], 0))
    out_path = args.out or (args.root / "outputs" / args.scene / "scene_with_trajectory.ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(out_path, out)
    print(f"[bake] → {out_path}  ({out['means'].shape[0]} gaussians, {out_path.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
