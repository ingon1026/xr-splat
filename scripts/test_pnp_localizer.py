"""test_pnp_localizer.py — PnPLocalizer 자가검증.

KF 몇 개를 query로 hint 없이 전역 relocalize 수행:
1. 추정 Tcw vs COLMAP Tcw 포즈오차 (cm / °)
2. 1프레임 localize 소요시간 (ms) → 추정 FPS
3. inlier 수 · confidence
"""
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.backproject import read_colmap_images, colmap_world_RT  # noqa: E402
from scripts.pnp_localizer import PnPLocalizer                         # noqa: E402
from pipeline.runtime import OK                                         # noqa: E402

DATA_DIR   = ROOT / "data/processed/ros2_bag2_home_rgbd_orbframe"
COLMAP_DIR = DATA_DIR / "colmap/sparse/0"
MAP_PATH   = ROOT / "outputs/ros2_bag2_home_rgbd_orbframe/feature_map.npz"

# 검증할 KF 수 (evenly spaced)
N_TEST = 10
# 포즈오차 통과 기준
MAX_TRANS_CM = 30.0   # cm
MAX_ROT_DEG  = 5.0    # degree


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """두 회전행렬 사이 각도(°)."""
    R_diff = R_est @ R_gt.T
    cos_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], dtype=np.float32)


def main():
    K = load_intrinsics(DATA_DIR / "intrinsics.json")
    kf_poses = read_colmap_images(COLMAP_DIR / "images.txt")

    if not MAP_PATH.exists():
        print(f"feature_map.npz 없음 — build_feature_map.py 먼저 실행하세요: {MAP_PATH}")
        sys.exit(1)

    localizer = PnPLocalizer(MAP_PATH, K)

    # 테스트할 KF 선택 (evenly spaced)
    kf_items = list(kf_poses.items())
    step = max(1, len(kf_items) // N_TEST)
    test_kfs = kf_items[::step][:N_TEST]

    print(f"{'KF':25s} {'trans_err(cm)':>14} {'rot_err(°)':>10} {'inliers':>8} {'conf':>6} {'ms':>8} {'state':>6}")
    print("-" * 85)

    trans_errs = []
    rot_errs   = []
    times_ms   = []
    n_ok       = 0

    for name, pose in test_kfs:
        rgb_path = DATA_DIR / "rgb" / name
        if not rgb_path.exists():
            print(f"{name:25s}  [rgb 없음]")
            continue

        # GT 포즈 (Tcw)
        qw, qx, qy, qz, tx, ty, tz = pose
        R_wc_gt, t_wc_gt = colmap_world_RT(qw, qx, qy, qz, tx, ty, tz)
        # Tcw: R_cw = R_wc^T, t_cw
        R_cw_gt = R_wc_gt.T
        t_cw_gt = -R_cw_gt @ t_wc_gt

        # query rgb 로드 (BGR→RGB)
        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # localize (hint 없음 — 전역)
        t0 = time.perf_counter()
        result = localizer.relocalize(rgb, hint=None)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        T_est = result.T_map_cam   # [4,4] Tcw
        R_est = T_est[:3, :3]
        t_est = T_est[:3, 3]

        # 포즈오차
        trans_err_m = float(np.linalg.norm(t_est - t_cw_gt))
        rot_err_deg = rotation_error_deg(R_est, R_cw_gt)

        trans_errs.append(trans_err_m * 100)   # cm
        rot_errs.append(rot_err_deg)
        times_ms.append(elapsed_ms)
        if result.state == OK:
            n_ok += 1

        n_inliers = localizer.last_n_inliers

        print(f"{name:25s}  {trans_err_m*100:14.1f}  {rot_err_deg:10.2f}  "
              f"{n_inliers:8d}  {result.confidence:6.3f}  {elapsed_ms:8.1f}  {result.state:>6}")

    print("-" * 85)
    if trans_errs:
        med_t   = float(np.median(trans_errs))
        med_r   = float(np.median(rot_errs))
        mean_ms = float(np.mean(times_ms))
        fps_est = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        print(f"중앙값 trans: {med_t:.1f} cm  |  중앙값 rot: {med_r:.2f}°")
        print(f"평균 {mean_ms:.1f} ms/frame  →  추정 {fps_est:.1f} FPS  (목표 30 FPS = 33 ms)")
        print(f"OK: {n_ok}/{len(trans_errs)}")

        real_time = mean_ms <= 33.3
        print()
        if real_time:
            print(f"[PASS] 실시간 가능 ({mean_ms:.1f} ms ≤ 33 ms)")
        else:
            print(f"[INFO] 실시간 미달 ({mean_ms:.1f} ms > 33 ms) — 최적화 여지 있음")

        trans_ok = med_t <= MAX_TRANS_CM
        rot_ok   = med_r <= MAX_ROT_DEG
        print(f"포즈오차 {'PASS' if trans_ok else 'FAIL'}: 중앙 trans {med_t:.1f} cm (기준 {MAX_TRANS_CM} cm)")
        print(f"포즈오차 {'PASS' if rot_ok   else 'FAIL'}: 중앙 rot   {med_r:.2f}° (기준 {MAX_ROT_DEG}°)")


if __name__ == "__main__":
    main()
