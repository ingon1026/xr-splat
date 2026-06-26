"""runtime_localizer.py — PhotometricLocalizer: pipeline/runtime.py Localizer 구현체.

가우시안 맵에 렌더를 쿼리 이미지와 맞춰 6DoF 포즈 최적화.
relocalize(거친 hint에서 150 iters) / track(prior에서 50 iters).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.runtime import Localizer, PoseResult, OK, LOST  # noqa: E402
from pipeline.gsplat_io import load_ply                        # noqa: E402
from photometric_reloc import se3_exp, render_at              # noqa: E402

# ── 신뢰도·상태 상수 (실측 기반: 회복 PSNR 22-27dB, lost PSNR <12dB) ──
PSNR_LOW  = 12.0   # conf=0 기준 (완전 틀린 포즈)
PSNR_HIGH = 25.0   # conf=1 기준 (잘 수렴한 포즈)
CONF_OK   = 0.4    # conf 임계: PSNR ≥ ~17.2 dB 면 OK


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """L2 기반 PSNR (dB). pred/gt 는 [H,W,3] float 0-1."""
    mse = F.mse_loss(pred.clamp(0, 1), gt).item()
    return float(-10.0 * np.log10(max(mse, 1e-10)))


def _conf_from_psnr(psnr: float) -> float:
    return float(np.clip((psnr - PSNR_LOW) / (PSNR_HIGH - PSNR_LOW), 0.0, 1.0))


class PhotometricLocalizer(Localizer):
    """Gaussian 맵 photometric 최적화 기반 위치추정.

    Args:
        ply_path: scene.ply 경로.
        K:        [3,3] torch.Tensor (float32, cuda) — 쿼리 이미지와 동일 해상도 기준.
        W, H:     쿼리 이미지 해상도 (K와 일치해야 함).
        device:   "cuda" 권장.
    """

    def __init__(self, ply_path, K: torch.Tensor, W: int, H: int, device: str = "cuda"):
        self.g = load_ply(ply_path, device)
        self.K = K
        self.W = W
        self.H = H
        self.device = device

    def _optimize(self, rgb: np.ndarray, vm_init: np.ndarray, iters: int):
        """vm_init 에서 se3 delta Adam 최적화.

        Args:
            rgb:     쿼리 이미지 [H,W,3] float 0-1 (W×H == self.W×self.H).
            vm_init: 초기 viewmat Tcw [4,4] float32.
            iters:   반복 횟수.

        Returns:
            (vm_final [4,4] np.float32, psnr float)
        """
        base = torch.tensor(vm_init, device=self.device, dtype=torch.float32)
        xi   = torch.zeros(6, device=self.device, requires_grad=True)
        opt  = torch.optim.Adam([xi], lr=3e-3)
        gt   = torch.tensor(rgb.copy(), device=self.device)

        for _ in range(iters):
            vm   = se3_exp(xi) @ base
            pred = render_at(self.g, vm, self.K, self.W, self.H).clamp(0, 1)
            loss = F.l1_loss(pred, gt)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            vm_final = (se3_exp(xi) @ base).detach().cpu().numpy().astype(np.float32)
            render   = render_at(
                self.g, torch.tensor(vm_final, device=self.device), self.K, self.W, self.H
            ).clamp(0, 1)
            psnr = _psnr(render, gt)

        return vm_final, psnr

    def relocalize(self, rgb: np.ndarray, hint: Optional[np.ndarray] = None) -> PoseResult:
        """전역/초기/복구 위치추정.

        hint 있으면 그 근방 150 iters 최적화.
        hint 없으면(전역 place-recognition 범위 밖) LOST 반환.
        """
        if hint is None:
            return PoseResult(
                T_map_cam=np.eye(4, dtype=np.float32),
                state=LOST,
                confidence=0.0,
            )

        vm_final, psnr = self._optimize(rgb, hint, iters=150)
        conf  = _conf_from_psnr(psnr)
        state = OK if conf >= CONF_OK else LOST
        return PoseResult(T_map_cam=vm_final, state=state, confidence=conf)

    def track(self, rgb: np.ndarray, prior: PoseResult) -> PoseResult:
        """이전 포즈(prior)에서 50 iters 추적 (프레임간 빠른 경로)."""
        vm_final, psnr = self._optimize(rgb, prior.T_map_cam, iters=50)
        conf  = _conf_from_psnr(psnr)
        state = OK if conf >= CONF_OK else LOST
        return PoseResult(T_map_cam=vm_final, state=state, confidence=conf)
