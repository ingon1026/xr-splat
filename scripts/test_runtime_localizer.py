"""test_runtime_localizer.py — PhotometricLocalizer 자가검증.

테스트 3가지:
  1. relocalize: 알려진 KF 포즈를 5cm/3° 섭동 → hint 로 relocalize → 포즈 회복 + conf 높음
  2. track     : 인접 두 KF, 첫 포즈로 둘째 track → state OK + 합리적 포즈
  3. (음성)    : 먼 엉뚱한 포즈를 prior로 track → conf 낮음·state LOST
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from pipeline.runtime import OK, LOST                                       # noqa: E402

import pytest  # noqa: E402
pytest.importorskip("gsplat")  # gsplat(CUDA) 없는 환경에선 이 모듈만 skip — 순수 테스트는 계속 동작
from runtime_localizer import PhotometricLocalizer                          # noqa: E402

# ── 자산 경로 ──
SCENE   = "ros2_bag2_home_rgbd_orbframe"
TAG     = "mcmc2m"
PROC    = ROOT / "data/processed" / SCENE
GDIR    = ROOT / "outputs" / SCENE / f"gsplat_{TAG}"
SP      = PROC / "colmap/sparse/0"
PLY     = GDIR / "scene.ply"
SCALE   = 0.5   # photometric_reloc 와 동일

DEV = "cuda"
RNG = np.random.default_rng(42)


def build_viewmat(qw, qx, qy, qz, tx, ty, tz) -> np.ndarray:
    """COLMAP Tcw → 4x4 viewmat."""
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    vm[:3, 3]  = [tx, ty, tz]
    return vm


def pose_err(vm_a: np.ndarray, vm_b: np.ndarray):
    """두 viewmat 사이 카메라 중심거리(cm)·회전각(deg)."""
    Ca, Cb = np.linalg.inv(vm_a), np.linalg.inv(vm_b)
    dt = np.linalg.norm(Ca[:3, 3] - Cb[:3, 3]) * 100
    dr = np.rad2deg(np.linalg.norm(
        Rot.from_matrix(Cb[:3, :3] @ Ca[:3, :3].T).as_rotvec()))
    return dt, dr


def perturb(vm_ref: np.ndarray, trans_m: float, rot_deg: float) -> np.ndarray:
    """카메라 중심 이동 + 회전 섭동."""
    C = np.linalg.inv(vm_ref).astype(np.float64)
    d = RNG.normal(size=3); d /= np.linalg.norm(d)
    C[:3, 3] += d * trans_m
    ax = RNG.normal(size=3); ax /= np.linalg.norm(ax)
    C[:3, :3] = Rot.from_rotvec(ax * np.deg2rad(rot_deg)).as_matrix() @ C[:3, :3]
    return np.linalg.inv(C).astype(np.float32)


def load_img(name: str, W: int, H: int) -> np.ndarray:
    img = cv2.imread(str(PROC / "rgb" / name))
    if img is None:
        raise FileNotFoundError(f"이미지 없음: {PROC / 'rgb' / name}")
    return cv2.resize(img[:, :, ::-1].astype(np.float32) / 255.0, (W, H))


def main():
    # ── 환경 초기화 ──
    W0, H0, fx, fy, cx, cy = read_colmap_cameras(SP / "cameras.txt")
    s  = SCALE
    W, H = int(W0 * s), int(H0 * s)
    K = torch.tensor([[fx*s, 0, cx*s], [0, fy*s, cy*s], [0, 0, 1.0]], device=DEV)

    imgs = read_colmap_images(SP / "images.txt")
    holdout = [l.strip() for l in open(GDIR / "holdout.txt") if l.strip() and l.strip() in imgs]
    holdout.sort()
    assert len(holdout) >= 3, "holdout 프레임 부족"

    print(f"[setup] W×H={W}×{H}, holdout={len(holdout)} frames")
    print(f"[setup] PLY={PLY}")

    loc = PhotometricLocalizer(PLY, K, W, H, device=DEV)

    pass_count = 0
    fail_count = 0

    # ── 테스트 1: relocalize (5cm/3° 섭동 hint) ──
    print("\n=== Test 1: relocalize (hint=섭동 5cm/3°) ===")
    n0 = holdout[0]
    vm_ref = build_viewmat(*imgs[n0])
    vm_hint = perturb(vm_ref, trans_m=0.05, rot_deg=3.0)
    rgb0 = load_img(n0, W, H)

    dt_init, dr_init = pose_err(vm_hint, vm_ref)
    result1 = loc.relocalize(rgb0, hint=vm_hint)
    dt_fin,  dr_fin  = pose_err(result1.T_map_cam, vm_ref)

    print(f"  frame : {n0}")
    print(f"  포즈오차 init: {dt_init:.2f}cm / {dr_init:.2f}°")
    print(f"  포즈오차 final: {dt_fin:.2f}cm / {dr_fin:.2f}°")
    print(f"  state={result1.state}  conf={result1.confidence:.3f}")

    t1_pass = (result1.state == OK and result1.confidence >= 0.4
               and dt_fin < dt_init and dr_fin < dr_init)
    if t1_pass:
        print("  [PASS] relocalize 포즈 회복 + conf OK")
        pass_count += 1
    else:
        print("  [FAIL] relocalize 기대 미달")
        fail_count += 1

    # ── 테스트 2: track (연속 COLMAP 프레임, 1-2cm 간격) ──
    print("\n=== Test 2: track (연속 COLMAP 프레임, 명목 inter-frame 모션) ===")
    from pipeline.runtime import PoseResult

    # holdout이 아닌 전체 이미지에서 타임스탬프 정렬 후 가장 가까운 연속 쌍 선택
    all_names = sorted(imgs.keys())
    n1, n2 = all_names[0], all_names[1]
    vm_ref1 = build_viewmat(*imgs[n1])
    vm_ref2 = build_viewmat(*imgs[n2])
    rgb2 = load_img(n2, W, H)

    # prior: 첫 프레임 GT 포즈
    prior = PoseResult(T_map_cam=vm_ref1, state=OK, confidence=1.0)

    dt_ref, dr_ref = pose_err(vm_ref1, vm_ref2)
    print(f"  frames: {n1} → {n2}")
    print(f"  두 프레임 사이 실제 모션: {dt_ref:.2f}cm / {dr_ref:.2f}°")

    result2 = loc.track(rgb2, prior)
    dt_track, dr_track = pose_err(result2.T_map_cam, vm_ref2)
    # prior에서 GT까지 거리 — track 후 이보다 줄어야 "수렴 방향 맞음"
    dt_prior_vs_gt, _ = pose_err(vm_ref1, vm_ref2)
    print(f"  track 포즈오차 vs GT: {dt_track:.2f}cm / {dr_track:.2f}°")
    print(f"  state={result2.state}  conf={result2.confidence:.3f}")

    # 통과: state OK + conf >= 0.4 + prior보다 GT에 가까워졌거나 이미 sub-cm
    t2_pass = (result2.state == OK and result2.confidence >= 0.4
               and dt_track < dt_prior_vs_gt)
    if t2_pass:
        print("  [PASS] track state OK + 포즈 GT 방향으로 수렴")
        pass_count += 1
    else:
        print("  [FAIL] track 기대 미달")
        fail_count += 1

    # ── 테스트 3: (음성) 엉뚱한 먼 포즈 → LOST ──
    print("\n=== Test 3: (음성) 먼 prior (>50cm) → LOST ===")
    n3 = holdout[0]
    vm_ref3 = build_viewmat(*imgs[n3])
    # convergence basin(~5cm) 밖으로 크게 이탈: 80cm 이동
    vm_far = perturb(vm_ref3, trans_m=0.80, rot_deg=45.0)
    rgb3 = load_img(n3, W, H)

    dt_far, dr_far = pose_err(vm_far, vm_ref3)
    print(f"  frame : {n3}")
    print(f"  far prior 오차: {dt_far:.2f}cm / {dr_far:.2f}°")

    prior_far = PoseResult(T_map_cam=vm_far, state=OK, confidence=0.9)
    result3 = loc.track(rgb3, prior_far)
    print(f"  state={result3.state}  conf={result3.confidence:.3f}")

    # 음성: conf < 0.4 이거나 state LOST 이면 PASS
    t3_pass = result3.state == LOST or result3.confidence < 0.4
    if t3_pass:
        print("  [PASS] 먼 포즈 → 낮은 conf / LOST (올바른 신호)")
        pass_count += 1
    else:
        print("  [WARN] 먼 포즈인데 conf 높음 — convergence basin이 예상보다 넓거나 임계 조정 필요")
        fail_count += 1

    # ── 최종 판정 ──
    print(f"\n{'='*50}")
    print(f"결과: {pass_count}/3 PASS  {fail_count}/3 FAIL")
    if fail_count == 0:
        print("전체 PASS")
    else:
        print("일부 FAIL — 위 수치 확인 후 임계 조정 필요")
    return fail_count == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
