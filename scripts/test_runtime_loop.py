#!/usr/bin/env python3
"""test_runtime_loop.py — run_loop 상태기계 단위 검증 (GPU 불필요, assert 기반).

검증 케이스:
  (a) 첫 프레임은 반드시 relocalize 경로
  (b) OK이고 conf > conf_thresh 이면 다음 프레임은 track
  (c) conf < conf_thresh 이면 다음 프레임은 relocalize (low-conf recovery)
  (d) LOST 상태이면 render=None, 그리고 다음 프레임도 relocalize
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.runtime import Localizer, PoseResult, OK, LOST, run_loop  # noqa: E402


# ---------------------------------------------------------------------------
# MockLocalizer — 결정론적, 호출 경로 기록
# ---------------------------------------------------------------------------

class MockLocalizer(Localizer):
    """conf_schedule에 따라 PoseResult 반환. 호출이 relocalize인지 track인지 self.calls에 기록."""

    def __init__(self, conf_schedule: list, state_schedule: list = None, rng_seed: int = 0):
        """
        conf_schedule: 프레임 i의 confidence 값 목록.
        state_schedule: 프레임 i의 state ("OK"/"LOST"). None이면 conf>0.15를 OK로.
        """
        self._conf = conf_schedule
        self._state = state_schedule
        self._rng = np.random.default_rng(rng_seed)
        self._idx = 0
        self.calls = []  # [("relocalize"|"track", frame_idx)]

    def _make_pose(self, method: str) -> PoseResult:
        i = self._idx
        self._idx += 1
        self.calls.append((method, i))
        conf = float(self._conf[i]) if i < len(self._conf) else 0.7
        if self._state is not None and i < len(self._state):
            state = self._state[i]
        else:
            state = OK if conf > 0.15 else LOST
        # 단위행렬에 작은 노이즈 (CPU only)
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self._rng.normal(0, 0.01, 3)
        return PoseResult(T_map_cam=T, state=state, confidence=conf)

    def relocalize(self, rgb: np.ndarray, hint=None) -> PoseResult:
        return self._make_pose("relocalize")

    def track(self, rgb: np.ndarray, prior: PoseResult) -> PoseResult:
        return self._make_pose("track")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def dummy_rgb():
    """렌더 불필요 — 빈 배열."""
    return np.zeros((4, 4, 3), dtype=np.float32)


def noop_render(T):
    """CPU 더미 렌더 — 항상 OK 반환(배열)."""
    return np.zeros((4, 4, 3), dtype=np.float32)


def make_frames(n):
    return [dummy_rgb() for _ in range(n)]


# ---------------------------------------------------------------------------
# 케이스 (a): 첫 프레임은 relocalize
# ---------------------------------------------------------------------------

def test_a_first_frame_is_relocalize():
    loc = MockLocalizer(conf_schedule=[0.8, 0.8, 0.8])
    results = run_loop(loc, make_frames(3), noop_render, conf_thresh=0.5)
    assert results[0]["idx"] == 0
    assert loc.calls[0] == ("relocalize", 0), f"첫 프레임이 relocalize여야 함: {loc.calls[0]}"
    print("(a) PASS: 첫 프레임 → relocalize")


# ---------------------------------------------------------------------------
# 케이스 (b): OK & conf > thresh 이면 다음 프레임은 track
# ---------------------------------------------------------------------------

def test_b_ok_high_conf_then_track():
    # frame 0: relocalize → OK, conf=0.8
    # frame 1: prev OK conf=0.8 > 0.5 → track
    # frame 2: prev OK conf=0.8 > 0.5 → track
    loc = MockLocalizer(conf_schedule=[0.8, 0.8, 0.8])
    run_loop(loc, make_frames(3), noop_render, conf_thresh=0.5)
    assert loc.calls[0][0] == "relocalize", "frame 0 must be relocalize"
    assert loc.calls[1][0] == "track", f"frame 1 should be track (prev OK, conf=0.8): {loc.calls[1]}"
    assert loc.calls[2][0] == "track", f"frame 2 should be track: {loc.calls[2]}"
    print("(b) PASS: OK+고신뢰 이후 → track")


# ---------------------------------------------------------------------------
# 케이스 (c): conf < conf_thresh 이면 다음 프레임이 relocalize
# ---------------------------------------------------------------------------

def test_c_low_conf_triggers_relocalize():
    # frame 0: relocalize → OK, conf=0.8
    # frame 1: track → OK, conf=0.3 (< 0.5)
    # frame 2: prev.conf=0.3 < 0.5 → relocalize
    # frame 3: track (conf 복구=0.8)
    conf = [0.8, 0.8, 0.3, 0.8, 0.8]
    loc = MockLocalizer(conf_schedule=conf)
    run_loop(loc, make_frames(5), noop_render, conf_thresh=0.5)

    # calls 순서: relocalize(0), track(1), track(2), relocalize(3), track(4)
    # 왜 track(2)? frame 1 결과는 conf=0.8>0.5 → frame 2는 track
    # frame 2 결과는 conf=0.3 → frame 3는 relocalize
    assert loc.calls[0][0] == "relocalize", f"frame 0: {loc.calls[0]}"
    assert loc.calls[1][0] == "track",      f"frame 1: {loc.calls[1]}"
    assert loc.calls[2][0] == "track",      f"frame 2: {loc.calls[2]}"
    assert loc.calls[3][0] == "relocalize", (
        f"frame 3 이전 conf=0.3<0.5 이므로 relocalize여야 함: {loc.calls[3]}")
    assert loc.calls[4][0] == "track",      f"frame 4: {loc.calls[4]}"
    print("(c) PASS: conf<thresh → 다음 프레임 relocalize")


# ---------------------------------------------------------------------------
# 케이스 (d): LOST 이면 render=None + 다음도 relocalize
# ---------------------------------------------------------------------------

def test_d_lost_render_none_then_relocalize():
    # frame 0: relocalize → OK, conf=0.8
    # frame 1: track → LOST (conf=0.05)
    # frame 2: prev LOST → relocalize
    conf   = [0.8, 0.05, 0.8]
    states = [OK,  LOST,  OK]
    loc = MockLocalizer(conf_schedule=conf, state_schedule=states)
    results = run_loop(loc, make_frames(3), noop_render, conf_thresh=0.5)

    # LOST 프레임 render=None
    assert results[1]["render"] is None, (
        f"LOST 프레임 render는 None이어야 함: {results[1]['render']}")
    # OK 프레임은 render 있음
    assert results[0]["render"] is not None, "OK 프레임 render 있어야 함"
    # 다음 프레임은 relocalize
    assert loc.calls[2][0] == "relocalize", (
        f"LOST 다음 프레임은 relocalize여야 함: {loc.calls[2]}")
    print("(d) PASS: LOST → render=None, 다음 relocalize")


# ---------------------------------------------------------------------------
# 보너스: LOST 상태에서도 state machine이 올바른 PoseResult를 out에 포함
# ---------------------------------------------------------------------------

def test_e_lost_pose_still_in_output():
    conf   = [0.8, 0.05]
    states = [OK,  LOST]
    loc = MockLocalizer(conf_schedule=conf, state_schedule=states)
    results = run_loop(loc, make_frames(2), noop_render, conf_thresh=0.5)

    assert results[1]["pose"].state == LOST
    assert results[1]["idx"] == 1
    print("(e) PASS: LOST PoseResult도 out에 포함")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_a_first_frame_is_relocalize,
        test_b_ok_high_conf_then_track,
        test_c_low_conf_triggers_relocalize,
        test_d_lost_render_none_then_relocalize,
        test_e_lost_pose_still_in_output,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL [{t.__name__}]: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR [{t.__name__}]: {type(e).__name__}: {e}")
            failed += 1

    print()
    if failed == 0:
        print(f"모든 테스트 통과 ({len(tests)}/{len(tests)})")
    else:
        print(f"{failed}/{len(tests)} 실패")
        sys.exit(1)
