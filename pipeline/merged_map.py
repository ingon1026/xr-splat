"""merged_map.py — MergedMap: 런타임 API 단일 진입점.

맵 로드·localize·render를 하나의 객체로 묶는다.
설계: compose, not rewrite — 기존 인터페이스·구현 재사용.

렌더는 항상 풀 자산(cfg.asset_ply, GPU). lite/저해상도 렌더 금지.

주의: PhotometricLocalizer는 hint=None이면 LOST 반환(전역 place-recognition 범위 밖).
  localize()는 내부적으로 relocalize(rgb, hint=None)을 부른다.
  photometric 모드에서 첫 호출은 외부에서 hint를 줘야 한다(SeededLocalizer 패턴 참고).
  pnp 모드는 전역 localize 가능하므로 hint 없이도 동작한다.

gsplat 의존: 이 파일을 import만 해도 gsplat 불필요.
  gsplat는 __init__ / render 내부에서 lazy import.
  테스트: pytest.importorskip("gsplat").
"""
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.config import load_config, SceneConfig  # noqa: E402
from pipeline.backproject import read_colmap_cameras   # noqa: E402
from pipeline.runtime import Localizer, PoseResult, run_loop  # noqa: E402


def _load_module(name: str, path: Path):
    """scripts/ 아래 모듈을 importlib로 안전하게 로드."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MergedMap:
    """런타임 API: 맵 로드 → localize → render → run.

    Args:
        scene_or_cfg: scene 이름 또는 SceneConfig 인스턴스.
        localizer:    "pnp"(기본, CPU, 전역) | "photometric"(GPU, hint 필요).
        device:       GPU 장치 ("cuda").
    """

    def __init__(self, scene_or_cfg, localizer: str = "pnp", device: str = "cuda"):
        # ── config 로드 ────────────────────────────────────────────────────
        if isinstance(scene_or_cfg, SceneConfig):
            cfg = scene_or_cfg
        else:
            cfg = load_config(scene_or_cfg)
        self.cfg = cfg
        self.device = device

        # ── 카메라 내재 파라미터 ────────────────────────────────────────────
        cameras_txt = cfg.sparse_dir / "cameras.txt"
        W, H, fx, fy, cx, cy = read_colmap_cameras(str(cameras_txt))
        self.W = W
        self.H = H
        self.K_np = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        # ── 풀 자산 로드 (GPU) — 렌더 전용 ─────────────────────────────────
        # gsplat_io lazy import: import pipeline.merged_map 자체는 gsplat 불필요
        import torch
        from pipeline.gsplat_io import load_ply  # noqa: E402

        self._g = load_ply(str(cfg.asset_ply), device=device)
        self.K_torch = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=device, dtype=torch.float32,
        )

        # ── Localizer 구성 ─────────────────────────────────────────────────
        self._localizer = self._build_localizer(localizer, cfg)

    # ── 내부 ──────────────────────────────────────────────────────────────

    def _build_localizer(self, kind: str, cfg: SceneConfig) -> Localizer:
        if kind == "pnp":
            mod = _load_module(
                "pnp_localizer", ROOT / "scripts" / "pnp_localizer.py"
            )
            return mod.PnPLocalizer(cfg.feature_map, self.K_np)

        elif kind == "photometric":
            import torch  # already imported above, but kept explicit
            mod = _load_module(
                "runtime_localizer", ROOT / "scripts" / "runtime_localizer.py"
            )
            return mod.PhotometricLocalizer(
                str(cfg.asset_ply), self.K_torch, self.W, self.H,
                device=self.device,
            )

        else:
            raise ValueError(f"알 수 없는 localizer: {kind!r}  (pnp | photometric)")

    # ── 공개 API ──────────────────────────────────────────────────────────

    def localize(self, rgb: np.ndarray, hint: Optional[np.ndarray] = None) -> PoseResult:
        """query rgb → PoseResult (T_map_cam [4,4], state OK|LOST, confidence 0-1).

        Args:
            rgb:  [H,W,3] float32 0-1 또는 uint8.
            hint: 거친 초기 Tcw [4,4] (photometric 모드 필수; pnp 모드 무관).
        """
        return self._localizer.relocalize(rgb, hint=hint)

    def render(self, T_map_cam: np.ndarray) -> np.ndarray:
        """Tcw 포즈 → 렌더 이미지 [H,W,3] float32 0-1 (풀 자산, 풀 해상도).

        항상 풀 자산(cfg.asset_ply) 사용. lite/저해상도 렌더 금지.
        """
        import torch
        from pipeline.gsplat_io import render as gs_render  # noqa: E402

        vm = torch.tensor(T_map_cam, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            rgb = gs_render(self._g, vm, self.K_torch, self.W, self.H)
        return rgb.cpu().numpy()

    def run(self, frames, conf_thresh: float = 0.5) -> list:
        """프레임 스트림을 위치추정→렌더 루프로 흘린다.

        Args:
            frames:      iterable of rgb [H,W,3] float 0-1.
            conf_thresh: confidence 임계값 (이하이면 다음 프레임 relocalize).

        Returns:
            list of dict(idx, pose:PoseResult, render: np.ndarray|None).
        """
        return run_loop(self._localizer, frames, self.render, conf_thresh)
