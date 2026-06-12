#!/usr/bin/env python3
"""04_make_pointcloud.py — N 키프레임 depth를 COLMAP 포즈(Tcw)로 월드 역투영 → 초기 포인트클라우드.

gsplat 초기화 입력. 산출: processed/<scene>/colmap/sparse/0/points3D.txt + colmap/points_init.ply.
좌표계는 COLMAP 모델과 동일(Tcw 역변환으로 월드). distortion 미보정(PINHOLE) — 가장자리 약간 부정확.

usage: 04_make_pointcloud.py --scene <scene> [--n 30] [--voxel 0.02]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.backproject import read_colmap_images, colmap_world_RT, backproject  # noqa: E402

log = logging.getLogger("pointcloud")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--n", type=int, default=30, help="사용할 키프레임 수")
    ap.add_argument("--voxel", type=float, default=0.02, help="voxel downsample [m]")
    ap.add_argument("--stride", type=int, default=4, help="픽셀 샘플 간격")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    proc = args.root / "data" / "processed" / args.scene
    sparse = proc / "colmap" / "sparse" / "0"
    intr = json.loads((proc / "intrinsics.json").read_text())
    fx, fy, cx, cy, ds = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["depth_scale"]
    imgs = read_colmap_images(sparse / "images.txt")            # name → Tcw
    assoc = {}                                                  # rgb basename → depth relpath
    for line in open(proc / "associations.txt"):
        p = line.split()
        if len(p) >= 4:
            assoc[Path(p[1]).name] = p[3]

    names = sorted(imgs.keys())
    step = max(1, len(names) // args.n)
    sel = names[::step][:args.n]
    log.info("키프레임 %d/%d 선택 (stride=%d, voxel=%.3f)", len(sel), len(names), args.stride, args.voxel)

    worlds, colors = [], []
    for name in sel:
        dpath = proc / assoc.get(name, "")
        if not dpath.exists():
            log.warning("depth 없음: %s", name); continue
        depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED).astype(np.float32) / ds
        rgb = cv2.imread(str(proc / "rgb" / name))
        R_wc, t_wc = colmap_world_RT(*imgs[name])
        w, _, c = backproject(depth, fx, fy, cx, cy, R_wc, t_wc, stride=args.stride, rgb=rgb)
        worlds.append(w); colors.append(c)
    if not worlds:
        log.error("역투영된 점 0 → 중단"); sys.exit(1)

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(worlds))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    pcd = pcd.voxel_down_sample(args.voxel)
    o3d.io.write_point_cloud(str(proc / "colmap" / "points_init.ply"), pcd)

    pts = np.asarray(pcd.points)
    cols = np.clip(np.asarray(pcd.colors) * 255, 0, 255).astype(int)
    with open(sparse / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, co) in enumerate(zip(pts, cols), start=1):
            f.write(f"{i} {p[0]} {p[1]} {p[2]} {co[0]} {co[1]} {co[2]} 0\n")
    log.info("초기 포인트클라우드: %d pts → points3D.txt + points_init.ply", len(pts))


if __name__ == "__main__":
    main()
