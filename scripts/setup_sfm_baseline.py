#!/usr/bin/env python3
"""setup_sfm_baseline.py — Phase 6 베이스라인: COLMAP SfM 포즈를 metric으로 스케일 정렬 → 베이스라인 scene.

COLMAP SfM는 스케일 임의. metric depth와 일치시키기 위해:
  s = median(metric_depth / COLMAP_sparse_depth)  (sparse 관측, 유효 depth 픽셀만)
  비율 분포가 tight한지 확인(IQR/median 리포트) — wide면 스케일 드리프트/매칭 불량 신호.
  s를 포즈 translation(과 포인트)에 적용.
출력: data/processed/<scene>_sfm/ (scaled COLMAP 포즈 + rgb/depth/intrinsics/associations 공유) → 04+06로 학습.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot


def read_images_with_obs(path):
    data = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    out = {}
    for i in range(0, len(data) - 1, 2):
        s = data[i].split()
        if len(s) < 10:
            continue
        name = s[9]
        pose = tuple(float(x) for x in s[1:8])   # qw qx qy qz tx ty tz (Tcw)
        o = data[i + 1].split()
        obs = [(float(o[j]), float(o[j + 1]), int(o[j + 2])) for j in range(0, len(o) - 2, 3)]
        out[name] = (pose, obs)
    return out


def read_points3d_xyz(path):
    pts = {}
    for l in open(path):
        if l.startswith("#"):
            continue
        s = l.split()
        if len(s) >= 4:
            pts[int(s[0])] = np.array([float(s[1]), float(s[2]), float(s[3])])
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    proc = args.root / "data" / "processed" / args.scene
    sfm = args.root / "outputs" / args.scene / "colmap_sfm" / "0"
    intr = json.loads((proc / "intrinsics.json").read_text())
    fx, fy, cx, cy, ds = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["depth_scale"]
    W, H = int(intr["width"]), int(intr["height"])
    imgs = read_images_with_obs(sfm / "images.txt")
    pts = read_points3d_xyz(sfm / "points3D.txt")
    assoc = {Path(p[1]).name: p[3] for p in (l.split() for l in open(proc / "associations.txt")) if len(p) >= 4}

    ratios = []
    for name, (pose, obs) in imgs.items():
        qw, qx, qy, qz, tx, ty, tz = pose
        Rcw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        drel = assoc.get(name)
        if not drel:
            continue
        depth = cv2.imread(str(proc / drel), cv2.IMREAD_UNCHANGED).astype(np.float32) / ds
        for x, y, pid in obs:
            if pid < 0 or pid not in pts:
                continue
            cdep = (Rcw @ pts[pid] + t)[2]          # COLMAP 카메라프레임 깊이
            u, v = int(round(x)), int(round(y))
            if cdep > 0 and 0 <= u < W and 0 <= v < H:
                md = depth[v, u]                    # 센서 metric 깊이
                if md > 0:
                    ratios.append(md / cdep)
    ratios = np.array(ratios)
    if len(ratios) < 50:
        print(f"[align] 유효 비율 {len(ratios)}개 — 너무 적음, 중단"); sys.exit(1)
    s = float(np.median(ratios))
    p25, p75 = np.percentile(ratios, [25, 75])
    spread = (p75 - p25) / s
    print(f"[align] ratio n={len(ratios)} median(s)={s:.4f} IQR/median={spread:.3f} "
          f"{'tight ✅' if spread < 0.1 else '⚠ WIDE — 스케일 드리프트/매칭 의심'}")

    out = args.root / "data" / "processed" / f"{args.scene}_sfm"
    sp = out / "colmap" / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    for f in ["rgb", "depth", "intrinsics.json", "rgb.txt", "depth.txt", "associations.txt"]:
        link = out / f
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((proc / f).resolve())
    imglink = out / "colmap" / "images"
    if imglink.is_symlink() or imglink.exists():
        imglink.unlink()
    imglink.symlink_to((proc / "rgb").resolve())

    (sp / "cameras.txt").write_text(
        f"# Camera list\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    with open(sp / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#\n")
        for i, (name, (pose, _)) in enumerate(imgs.items(), 1):
            qw, qx, qy, qz, tx, ty, tz = pose
            f.write(f"{i} {qw} {qx} {qy} {qz} {s*tx} {s*ty} {s*tz} 1 {name}\n\n")  # translation × s
    (sp / "points3D.txt").write_text(
        "# 3D point list (Phase4 04_make_pointcloud.py가 metric depth로 채움)\n")
    print(f"[align] 베이스라인 scene → {out}  ({len(imgs)} 포즈, scale={s:.4f})")


if __name__ == "__main__":
    main()
