"""build_feature_map.py — ORB 특징점 기반 오프라인 feature map 빌드.

KF(images.txt 포즈 있는 프레임)마다 ORB 추출 → depth backproject → world 좌표 누적.
저장: outputs/ros2_bag2_home_rgbd_orbframe/feature_map.npz
  points3d: float32 [N,3] — world 좌표 3D 점
  desc:     uint8  [N,32] — 각 점의 ORB descriptor
  K:        float32 [3,3] — 빌드 시 intrinsics (참고용)
"""
import argparse
import sys
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.backproject import read_colmap_images, colmap_world_RT  # noqa: E402

N_FEATURES = 1000   # ORB keypoints per frame


def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    return d["fx"], d["fy"], d["cx"], d["cy"], d["depth_scale"]


def load_associations(path):
    """associations.txt → {rgb_stem: depth_path}."""
    assoc = {}
    for line in open(path):
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        # 키는 rgb 파일명 stem (COLMAP images.txt name과 매칭). parts[0]=timestamp는
        # 파일명≠timestamp인 scene(Replica: 000000.jpg vs ts 0.000000)에서 불일치 → 전 KF skip.
        rgb_stem  = Path(parts[1]).stem   # "rgb/000000.jpg" → "000000"
        depth_rel = parts[3]              # e.g. "depth/000000.png"
        assoc[rgb_stem] = depth_rel
    return assoc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    args = ap.parse_args()
    DATA_DIR = ROOT / "data/processed" / args.scene
    COLMAP_DIR = DATA_DIR / "colmap/sparse/0"
    OUT_PATH = ROOT / "outputs" / args.scene / "feature_map.npz"

    # intrinsics
    fx, fy, cx, cy, depth_scale = load_intrinsics(DATA_DIR / "intrinsics.json")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # KF poses: {name: (qw,qx,qy,qz,tx,ty,tz)}
    kf_poses = read_colmap_images(COLMAP_DIR / "images.txt")

    # associations: rgb_stem → depth relative path
    assoc = load_associations(DATA_DIR / "associations.txt")

    # ORB detector
    orb = cv2.ORB_create(nfeatures=N_FEATURES)

    all_pts3d = []
    all_desc  = []
    n_kf_used = 0
    n_kf_skip = 0

    for name, pose in kf_poses.items():
        stem = Path(name).stem   # "68.021077.png" → "68.021077"
        rgb_path   = DATA_DIR / "rgb" / name
        depth_stem = assoc.get(stem)
        if depth_stem is None:
            n_kf_skip += 1
            continue
        depth_path = DATA_DIR / depth_stem

        if not rgb_path.exists() or not depth_path.exists():
            n_kf_skip += 1
            continue

        rgb   = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)  # uint16
        if rgb is None or depth is None:
            n_kf_skip += 1
            continue

        # ORB 추출 (grayscale)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kps, descs = orb.detectAndCompute(gray, None)
        if descs is None or len(kps) == 0:
            n_kf_skip += 1
            continue

        # 포즈: Tcw → R_wc, t_wc
        qw, qx, qy, qz, tx, ty, tz = pose
        R_wc, t_wc = colmap_world_RT(qw, qx, qy, qz, tx, ty, tz)

        # 각 keypoint backproject
        for kp, desc in zip(kps, descs):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if u < 0 or u >= depth.shape[1] or v < 0 or v >= depth.shape[0]:
                continue
            d_raw = depth[v, u]
            if d_raw == 0:
                continue
            d = float(d_raw) / depth_scale   # meters

            # cam 좌표
            x_c = (u - cx) * d / fx
            y_c = (v - cy) * d / fy
            cam_pt = np.array([x_c, y_c, d], dtype=np.float64)

            # world 좌표
            world_pt = R_wc @ cam_pt + t_wc
            all_pts3d.append(world_pt.astype(np.float32))
            all_desc.append(desc)

        n_kf_used += 1

    points3d = np.array(all_pts3d, dtype=np.float32)   # [N, 3]
    desc_arr  = np.array(all_desc,  dtype=np.uint8)    # [N, 32]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT_PATH), points3d=points3d, desc=desc_arr, K=K)

    print(f"KF 사용: {n_kf_used}  건너뜀: {n_kf_skip}")
    print(f"3D 점 수: {len(points3d):,}")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
