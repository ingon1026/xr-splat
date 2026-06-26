"""runtime.py — decoupled SLAM×Gaussian 런타임 루프 인터페이스 (재진입 경로의 시스템화).

흐름:  pose stream → Localizer → (T_map_cam, state, confidence) → Renderer(viewmat) → image.

설계 원칙:
- 렌더러는 항상 Tcw(viewmat 4x4)만 받는다 (decoupled 유지).
- Localizer 구현은 교체 가능(photometric / COLMAP-PnP / 학습형 ...). 이 모듈은 *계약*만 정의한다.
- 두 모드: relocalize(전역/초기/복구) + track(프레임간 빠른 경로).
- "찍은 공간 안"에서만 신뢰 가능(MERGED-MAP-RESULTS.md envelope) — confidence가 그 경계를 신호한다.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

OK = "OK"
LOST = "LOST"


@dataclass
class PoseResult:
    """위치추정 1회 결과."""
    T_map_cam: np.ndarray   # [4,4] Tcw(world→cam) viewmat. state==LOST면 직전 추정/단위행렬일 수 있음.
    state: str              # OK | LOST
    confidence: float       # 0..1 (높을수록 신뢰; 보통 render-vs-query 광도일치 기반)


class Localizer:
    """맵에 대한 위치추정 인터페이스. 구현체는 이 두 메서드를 채운다."""

    def relocalize(self, rgb: np.ndarray, hint: Optional[np.ndarray] = None) -> PoseResult:
        """전역/초기/복구 위치추정. hint=거친 초기 Tcw(있으면 그 근방 탐색). 실패 시 state=LOST."""
        raise NotImplementedError

    def track(self, rgb: np.ndarray, prior: "PoseResult") -> PoseResult:
        """이전 포즈(prior)에서 출발해 현재 프레임 위치추정(프레임간 빠른 경로)."""
        raise NotImplementedError


def run_loop(localizer: Localizer, frames, render_fn, conf_thresh: float = 0.5):
    """프레임 스트림을 위치추정→렌더로 흘린다. 상태기계: OK는 track, LOST/시작은 relocalize.

    frames: iterable of rgb [H,W,3] float 0-1.
    render_fn: callable(T_map_cam[4,4]) -> rendered rgb [H,W,3], 또는 None 반환 허용.
    반환: list of dict(idx, pose:PoseResult, render: rgb|None).
    """
    out = []
    prev: Optional[PoseResult] = None
    for i, rgb in enumerate(frames):
        if prev is None or prev.state == LOST or prev.confidence < conf_thresh:
            pose = localizer.relocalize(rgb, hint=(prev.T_map_cam if prev is not None else None))
        else:
            pose = localizer.track(rgb, prev)
        render = render_fn(pose.T_map_cam) if pose.state == OK else None
        out.append(dict(idx=i, pose=pose, render=render))
        prev = pose
    return out
