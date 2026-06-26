"""test_benchmark_pnp.py — benchmark_pnp 집계·포즈오차·run_loop 단위검증.

gsplat·GPU 불필요. assert 기반.
실행:
  python scripts/test_benchmark_pnp.py
  pytest scripts/test_benchmark_pnp.py -v
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.runtime import Localizer, PoseResult, OK, LOST, run_loop  # noqa: E402
from benchmark_pnp import MockPnP, aggregate, pose_err                  # noqa: E402


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def make_vm(trans=(0.0, 0.0, 0.0), rot_deg=0.0, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """간단한 viewmat Tcw 생성."""
    vm = np.eye(4, dtype=np.float32)
    vm[:3, 3] = trans
    if rot_deg != 0.0:
        vm[:3, :3] = Rot.from_rotvec(
            np.array(axis, dtype=float) * np.deg2rad(rot_deg)
        ).as_matrix().astype(np.float32)
    return vm


def dummy_rgb() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.float32)


def noop_render(T: np.ndarray) -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.float32)


# ── pose_err 단위검증 ─────────────────────────────────────────────────────────

def test_pose_err_identity():
    """동일 포즈 → 오차 0."""
    vm = make_vm(trans=(1.0, 2.0, 3.0), rot_deg=30.0, axis=(1.0, 0.0, 0.0))
    dt, dr = pose_err(vm, vm)
    assert dt < 1e-4, f"trans 오차 {dt:.6f} (expect ≈ 0)"
    assert dr < 1e-4, f"rot 오차 {dr:.6f} (expect ≈ 0)"
    print("(pose_err/identity) PASS")


def test_pose_err_known_translation():
    """Tcw에서 1m 이동 → 카메라 중심 100cm 차이."""
    # vm_a: R=I, t=(0,0,0) → camera center = -I@(0,0,0) = (0,0,0)
    # vm_b: R=I, t=(1,0,0) → camera center = -I@(1,0,0) = (-1,0,0)
    # 거리 = 1m = 100cm
    vm_a = make_vm(trans=(0.0, 0.0, 0.0))
    vm_b = make_vm(trans=(1.0, 0.0, 0.0))
    dt, dr = pose_err(vm_a, vm_b)
    assert abs(dt - 100.0) < 1e-2, f"expected 100cm, got {dt:.4f}"
    assert dr < 1e-3, f"rot 오차 {dr:.4f} (expect ≈ 0)"
    print(f"(pose_err/trans) PASS: {dt:.4f}cm")


def test_pose_err_known_rotation():
    """Y축 90° 회전 → 90°, 이동 없음."""
    vm_a = make_vm()
    vm_b = make_vm(rot_deg=90.0, axis=(0.0, 1.0, 0.0))
    dt, dr = pose_err(vm_a, vm_b)
    assert abs(dr - 90.0) < 1e-2, f"expected 90°, got {dr:.4f}"
    assert dt < 1e-3, f"trans 오차 {dt:.4f} (expect ≈ 0)"
    print(f"(pose_err/rot) PASS: {dr:.4f}°")


# ── aggregate 단위검증 ────────────────────────────────────────────────────────

def test_aggregate_basic():
    """4 records: 2 OK + 2 LOST. 집계 수치 검증."""
    records = [
        dict(ms=20.0, trans_cm=0.5, rot_deg=0.3, state=OK),
        dict(ms=30.0, trans_cm=1.0, rot_deg=0.6, state=OK),
        dict(ms=40.0, trans_cm=3.0, rot_deg=2.0, state=LOST),
        dict(ms=50.0, trans_cm=5.0, rot_deg=4.0, state=LOST),
    ]
    agg = aggregate(records)

    assert agg["n_frames"] == 4
    assert abs(agg["ms_median"] - 35.0) < 0.1, f"median ms {agg['ms_median']}"
    assert abs(agg["reloc_ok_rate"] - 0.5) < 1e-6, f"ok_rate {agg['reloc_ok_rate']}"
    # 35ms > 33.3ms → 30FPS 미달
    assert agg["fps_30_target"] is False, f"fps_30_target 예상 False: {agg['fps_30_target']}"
    # §2 허용오차: <1° AND ≤2cm
    # frame0 (0.3° < 1°, 0.5cm ≤ 2cm) → True
    # frame1 (0.6° < 1°, 1.0cm ≤ 2cm) → True
    # frame2 (2.0° ≥ 1°, 3.0cm > 2cm) → False
    # frame3 (4.0° ≥ 1°, 5.0cm > 2cm) → False
    assert abs(agg["tight_ok_frac"] - 0.5) < 1e-6, f"tight_frac {agg['tight_ok_frac']}"
    print(f"(aggregate/basic) PASS: ok_rate={agg['reloc_ok_rate']:.1f} "
          f"tight={agg['tight_ok_frac']:.1f} fps_target={agg['fps_30_target']}")


def test_aggregate_fps_target():
    """25ms → median 25ms ≤ 33ms → 30 FPS 달성."""
    records = [dict(ms=25.0, trans_cm=0.5, rot_deg=0.3, state=OK) for _ in range(5)]
    agg = aggregate(records)
    assert agg["fps_30_target"] is True, f"25ms는 30FPS 달성이어야 함: {agg}"
    assert abs(agg["fps_median"] - 40.0) < 0.5, f"fps_median {agg['fps_median']}"
    print(f"(aggregate/fps_target) PASS: fps_median={agg['fps_median']:.1f}")


def test_aggregate_perfect_ok():
    """전부 OK → reloc_ok_rate 1.0."""
    records = [dict(ms=15.0, trans_cm=0.2, rot_deg=0.1, state=OK) for _ in range(10)]
    agg = aggregate(records)
    assert agg["reloc_ok_rate"] == 1.0
    assert agg["tight_ok_frac"] == 1.0
    assert agg["fps_30_target"] is True
    print("(aggregate/perfect_ok) PASS")


# ── run_loop + MockPnP 통합검증 ───────────────────────────────────────────────

def test_run_loop_with_mock():
    """MockPnP(latency=0)로 run_loop → n개 결과, 모두 OK."""
    n = 10
    gt_poses = [("f.png", np.eye(4, dtype=np.float32)) for _ in range(n)]
    loc = MockPnP(gt_poses=gt_poses, latency_ms=0.0,
                  noise_trans_m=0.001, noise_rot_deg=0.1)

    results = run_loop(loc, [dummy_rgb() for _ in range(n)], noop_render, conf_thresh=0.5)

    assert len(results) == n, f"결과 {len(results)} (expect {n})"
    assert all(r["pose"].state == OK for r in results), \
        f"일부 LOST: {[r['pose'].state for r in results]}"
    print(f"(run_loop/mock) PASS: {n}프레임 모두 OK")


def test_run_loop_state_machine():
    """첫 프레임 relocalize → 이후 track (conf=0.9 > 0.5)."""

    class _TrackingMock(Localizer):
        def __init__(self):
            self.calls = []

        def relocalize(self, rgb: np.ndarray, hint=None) -> PoseResult:
            self.calls.append("relocalize")
            return PoseResult(np.eye(4, dtype=np.float32), OK, 0.9)

        def track(self, rgb: np.ndarray, prior: PoseResult) -> PoseResult:
            self.calls.append("track")
            return PoseResult(np.eye(4, dtype=np.float32), OK, 0.9)

    loc = _TrackingMock()
    run_loop(loc, [dummy_rgb() for _ in range(4)], noop_render, conf_thresh=0.5)

    assert loc.calls[0] == "relocalize", \
        f"첫 호출이 relocalize여야 함: {loc.calls[0]}"
    assert all(c == "track" for c in loc.calls[1:]), \
        f"이후는 track이어야 함: {loc.calls[1:]}"
    print(f"(run_loop/state_machine) PASS: calls={loc.calls}")


def test_run_loop_lost_render_none():
    """LOST 프레임은 render=None."""

    class _LostMock(Localizer):
        def __init__(self):
            self._call = 0

        def relocalize(self, rgb, hint=None):
            self._call += 1
            if self._call == 2:
                return PoseResult(np.eye(4, dtype=np.float32), LOST, 0.0)
            return PoseResult(np.eye(4, dtype=np.float32), OK, 0.9)

        def track(self, rgb, prior):
            return self.relocalize(rgb)

    loc = _LostMock()
    results = run_loop(loc, [dummy_rgb() for _ in range(3)], noop_render, conf_thresh=0.5)
    assert results[1]["pose"].state == LOST
    assert results[1]["render"] is None, "LOST 프레임은 render=None"
    print("(run_loop/lost_render_none) PASS")


# ── MockPnP 노이즈 검증 ───────────────────────────────────────────────────────

def test_mock_pnp_noise_within_bounds():
    """MockPnP 노이즈가 설정 허용오차 내에 있는지 (0.5cm / 0.3°)."""
    n = 20
    gt_poses = [("f.png", np.eye(4, dtype=np.float32)) for _ in range(n)]
    loc = MockPnP(gt_poses=gt_poses, noise_trans_m=0.005, noise_rot_deg=0.3,
                  latency_ms=0.0)
    errors = []
    for _ in range(n):
        res = loc.relocalize(dummy_rgb())
        dt, dr = pose_err(res.T_map_cam, np.eye(4, dtype=np.float32))
        errors.append((dt, dr))

    max_dt = max(e[0] for e in errors)
    max_dr = max(e[1] for e in errors)
    # noise_trans_m=0.005 → 카메라 중심 이동 최대 0.5cm
    assert max_dt < 2.0, f"trans 최대 {max_dt:.3f}cm (expect < 2cm)"
    assert max_dr < 2.0, f"rot 최대 {max_dr:.3f}° (expect < 2°)"
    print(f"(mock_pnp/noise) PASS: max_trans={max_dt:.3f}cm max_rot={max_dr:.3f}°")


def test_mock_pnp_state_ok():
    """MockPnP는 항상 OK + conf=0.9 반환."""
    gt_poses = [("f.png", make_vm()) for _ in range(5)]
    loc = MockPnP(gt_poses=gt_poses, latency_ms=0.0)
    for _ in range(5):
        r = loc.relocalize(dummy_rgb())
        assert r.state == OK
        assert abs(r.confidence - 0.9) < 1e-6
    print("(mock_pnp/state_ok) PASS")


# ── main ─────────────────────────────────────────────────────────────────────

TESTS = [
    test_pose_err_identity,
    test_pose_err_known_translation,
    test_pose_err_known_rotation,
    test_aggregate_basic,
    test_aggregate_fps_target,
    test_aggregate_perfect_ok,
    test_run_loop_with_mock,
    test_run_loop_state_machine,
    test_run_loop_lost_render_none,
    test_mock_pnp_noise_within_bounds,
    test_mock_pnp_state_ok,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL [{t.__name__}]: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR [{t.__name__}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print()
    if failed == 0:
        print(f"모든 테스트 통과 ({len(TESTS)}/{len(TESTS)})")
    else:
        print(f"{failed}/{len(TESTS)} 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
