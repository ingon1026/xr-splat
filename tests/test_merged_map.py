"""test_merged_map.py — MergedMap 스모크 테스트.

home scene: MergedMap 생성 → non-KF query 1~3프레임 localize → 포즈 OK + render shape + PSNR.
gsplat 없으면 스킵(importorskip).

실행:
    pytest tests/test_merged_map.py -v
  또는:
    python tests/test_merged_map.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# gsplat 없으면 테스트 전체 스킵
gsplat = pytest.importorskip("gsplat")

from pipeline.merged_map import MergedMap          # noqa: E402
from pipeline.config import load_config            # noqa: E402
from pipeline.backproject import read_colmap_images  # noqa: E402
from pipeline.runtime import OK                    # noqa: E402

SCENE = "ros2_bag2_home_rgbd_orbframe"
N_FRAMES = 3          # non-KF query 수
NONKF_TIME_RANGE = (68.0, 85.0)   # benchmark_pnp.py와 동일


# ── 픽스처 ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def merged_map():
    """MergedMap(pnp) 한 번만 생성 — 무거운 PLY 로드 공유."""
    return MergedMap(SCENE, localizer="pnp", device="cuda")


@pytest.fixture(scope="module")
def non_kf_frames(merged_map):
    """맵에 없는 non-KF RGB 프레임 최대 N_FRAMES개 로드.

    benchmark_pnp.load_non_kf_frames와 같은 소스(rgb/, NONKF_TIME_RANGE).
    GT 포즈 없음 — localize state와 render shape 검증용.
    """
    cfg = merged_map.cfg
    rgb_dir = cfg.proc_dir / "rgb"
    kf_names = set(read_colmap_images(str(cfg.sparse_dir / "images.txt")).keys())

    t0, t1 = NONKF_TIME_RANGE
    all_files = sorted(
        (p for p in rgb_dir.iterdir() if p.suffix == ".png"),
        key=lambda p: float(p.stem),
    )
    non_kf_paths = [
        p for p in all_files
        if p.name not in kf_names and t0 <= float(p.stem) <= t1
    ][:N_FRAMES]

    if not non_kf_paths:
        pytest.skip(f"non-KF 프레임 없음 ({rgb_dir}, {t0}~{t1}s)")

    W, H = merged_map.W, merged_map.H
    frames = []
    for p in non_kf_paths:
        img = cv2.imread(str(p))
        assert img is not None, f"imread 실패: {p}"
        img = cv2.resize(img, (W, H))
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        frames.append((p.name, rgb))

    return frames


# ── 테스트 ────────────────────────────────────────────────────────────────────

def test_merged_map_init(merged_map):
    """MergedMap이 올바른 속성을 들고 있는지 확인."""
    mm = merged_map
    assert mm.W > 0 and mm.H > 0, f"해상도 이상: {mm.W}x{mm.H}"
    assert mm.K_np.shape == (3, 3)
    assert mm._g is not None, "가우시안 PLY 로드 실패"


def test_localize_render_nonkf(merged_map, non_kf_frames):
    """non-KF query 프레임: localize → state OK + render shape."""
    mm = merged_map
    n_ok = 0
    psnrs = []
    times_ms = []

    print(f"\n{'frame':30s}  {'state':6s}  {'conf':6s}  {'PSNR(dB)':9s}  {'ms':8s}")
    print("-" * 70)

    for name, rgb in non_kf_frames:
        t0 = time.perf_counter()
        pose = mm.localize(rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

        # 포즈 상태 확인
        assert pose.T_map_cam.shape == (4, 4), f"T shape 이상: {pose.T_map_cam.shape}"
        assert pose.state in (OK, "LOST"), f"state 이상: {pose.state!r}"
        assert 0.0 <= pose.confidence <= 1.0

        # 렌더 (pose state 무관 — 항상 렌더, 검증용)
        t1 = time.perf_counter()
        rendered = mm.render(pose.T_map_cam)
        render_ms = (time.perf_counter() - t1) * 1000.0

        assert rendered.shape == (mm.H, mm.W, 3), \
            f"render shape 이상: {rendered.shape}, 기대 ({mm.H},{mm.W},3)"
        assert rendered.dtype == np.float32, f"dtype 이상: {rendered.dtype}"
        assert rendered.min() >= 0.0 and rendered.max() <= 1.0 + 1e-5

        # PSNR (localize 성공 시)
        if pose.state == OK:
            n_ok += 1
            mse = float(np.mean((rendered - rgb) ** 2))
            psnr = -10.0 * np.log10(max(mse, 1e-10))
            psnrs.append(psnr)
        else:
            psnr = float("nan")

        print(f"{name:30s}  {pose.state:6s}  {pose.confidence:6.3f}  "
              f"{psnr:9.2f}  {elapsed_ms + render_ms:8.1f}")

    print("-" * 70)
    mean_ms = float(np.mean(times_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    print(f"localize 평균: {mean_ms:.1f} ms  →  추정 {fps:.1f} FPS")
    if psnrs:
        print(f"PSNR (OK 프레임): mean={np.mean(psnrs):.2f} dB")
    print(f"OK: {n_ok}/{len(non_kf_frames)}")

    # 최소 1프레임 이상 OK여야 스모크 통과
    assert n_ok >= 1, (
        f"non-KF {len(non_kf_frames)}프레임 중 OK가 하나도 없음 — "
        f"feature_map 확인 필요: {merged_map.cfg.feature_map}"
    )
    if psnrs:
        # PSNR 합리적 범위 (PnP 기준 ~28 dB; 10 dB 하한은 너무 낮은 경우 검출)
        assert np.mean(psnrs) > 10.0, f"PSNR 너무 낮음: {np.mean(psnrs):.2f} dB"


# ── standalone 실행 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
