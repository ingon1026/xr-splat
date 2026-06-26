"""benchmark_pnp.py — PnPLocalizer 속도·정확도·전역 reloc 성공률 벤치마크.

사용:
  python scripts/benchmark_pnp.py --localizer mock             # MockPnP, KF 프레임
  python scripts/benchmark_pnp.py --localizer pnp              # 실제 PnP, KF 프레임 (상한)
  python scripts/benchmark_pnp.py --localizer pnp --frames non-kf  # non-KF query (실전)

출력:
  stdout : 집계 표
  JSON   : outputs/ros2_bag2_home_rgbd_orbframe/pnp_benchmark{_nonkf}.json
  PNG    : docs/assets/report/pnp_benchmark{_nonkf}.png
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from pipeline.runtime import Localizer, PoseResult, OK, LOST              # noqa: E402

# ── 자산 경로 ────────────────────────────────────────────────────────────────
SCENE   = "ros2_bag2_home_rgbd_orbframe"
PROC    = ROOT / "data/processed" / SCENE
SP      = PROC / "colmap/sparse/0"
OUT_DIR = ROOT / "outputs" / SCENE
FEAT    = OUT_DIR / "feature_map.npz"

N_FRAMES_DEFAULT   = 30
NONKF_TIME_RANGE   = (68.0, 85.0)  # non-KF 탐색 구간 (초)
INLIER_OK          = 20             # pnp_localizer.py 와 동기

# §2 허용오차 기준
THRESH_ROT_DEG  = 1.0   # ≤ 1°
THRESH_TRANS_CM = 2.0   # ≤ 2 cm
FPS_TARGET      = 30.0  # 33ms


# ── 포즈 오차 (test_runtime_localizer.py 와 동일 정의) ──────────────────────
def pose_err(vm_a: np.ndarray, vm_b: np.ndarray):
    """두 Tcw viewmat 사이 카메라 중심거리(cm) · 회전각(deg)."""
    Ca = np.linalg.inv(vm_a)
    Cb = np.linalg.inv(vm_b)
    dt = np.linalg.norm(Ca[:3, 3] - Cb[:3, 3]) * 100  # m → cm
    dr = np.rad2deg(np.linalg.norm(
        Rot.from_matrix(Cb[:3, :3] @ Ca[:3, :3].T).as_rotvec()))
    return float(dt), float(dr)


# ── MockPnP ──────────────────────────────────────────────────────────────────
class MockPnP(Localizer):
    """GT 포즈 + 작은 노이즈. latency_ms sleep으로 CPU 시간 모사.

    Args:
        gt_poses:     list of (name, viewmat Tcw [4,4] float32) — 프레임 순서와 동일.
        noise_trans_m: 카메라 중심 노이즈 (m). 기본 0.005 m = 0.5 cm.
        noise_rot_deg: 회전 노이즈 (°). 기본 0.3°.
        latency_ms:    sleep 시간 (ms). 기본 20.0 ms → 약 50 FPS 모사.
        rng_seed:      재현성.
    """

    def __init__(
        self,
        gt_poses: list,
        noise_trans_m: float = 0.005,
        noise_rot_deg: float = 0.3,
        latency_ms: float = 20.0,
        rng_seed: int = 42,
    ):
        self._poses   = gt_poses
        self._idx     = 0
        self._noise_t = noise_trans_m
        self._noise_r = np.deg2rad(noise_rot_deg)
        self._latency = latency_ms / 1000.0
        self._rng     = np.random.default_rng(rng_seed)

    def _next_pose(self) -> np.ndarray:
        _, vm = self._poses[self._idx % len(self._poses)]
        self._idx += 1
        return vm.astype(np.float32)

    def _perturb(self, vm: np.ndarray) -> np.ndarray:
        """카메라 중심 노이즈 + 소 회전 섭동."""
        C = np.linalg.inv(vm.astype(np.float64))
        d = self._rng.normal(size=3)
        n = np.linalg.norm(d)
        if n > 0:
            d /= n
        C[:3, 3] += d * self._noise_t
        ax = self._rng.normal(size=3)
        an = np.linalg.norm(ax)
        if an > 0:
            ax /= an
        C[:3, :3] = Rot.from_rotvec(ax * self._noise_r).as_matrix() @ C[:3, :3]
        return np.linalg.inv(C).astype(np.float32)

    # PnPLocalizer 와 동일 진단 속성
    last_n_inliers: int = 80

    def relocalize(self, rgb: np.ndarray, hint: Optional[np.ndarray] = None) -> PoseResult:
        if self._latency > 0:
            time.sleep(self._latency)
        vm = self._perturb(self._next_pose())
        return PoseResult(T_map_cam=vm, state=OK, confidence=0.9)

    def track(self, rgb: np.ndarray, prior: PoseResult) -> PoseResult:
        return self.relocalize(rgb)


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def load_frames(proc_dir: Path, images_dict: dict, n_frames: int, W: int, H: int):
    """시간순 n_frames 개 KF 이미지 사전 로드 + GT viewmat.

    반환: list of (name, rgb [H,W,3] float32 0-1, gt_vm [4,4] float32 Tcw)
    이미지는 타이밍 루프 전에 메모리에 올려 I/O 영향 제거.
    """
    rgb_dir = proc_dir / "rgb"
    names = sorted(images_dict.keys(), key=lambda n: float(n[:-4]))[:n_frames]

    frames = []
    for name in names:
        img = cv2.imread(str(rgb_dir / name))
        if img is None:
            raise FileNotFoundError(f"이미지 없음: {rgb_dir / name}")
        img = cv2.resize(img, (W, H))
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0

        qw, qx, qy, qz, tx, ty, tz = images_dict[name]
        R = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3,  3] = [tx, ty, tz]

        frames.append((name, rgb, vm))

    print(f"[load] {len(frames)} 프레임 로드 완료 (해상도 {W}×{H})")
    return frames


# ── non-KF 프레임 로드 ───────────────────────────────────────────────────────
def load_non_kf_frames(proc_dir: Path, images_dict: dict, n_frames: int,
                        W: int, H: int,
                        time_range: tuple = NONKF_TIME_RANGE):
    """맵에 없는 non-KF 프레임 사전 로드. GT 포즈 없음(None).

    반환: list of (name, rgb [H,W,3] float32 0-1, None)
    """
    rgb_dir  = proc_dir / "rgb"
    kf_names = set(images_dict.keys())
    t0, t1   = time_range

    all_names = sorted(
        (n for n in rgb_dir.iterdir() if n.suffix == ".png"),
        key=lambda p: float(p.stem),
    )
    non_kf = [
        p.name for p in all_names
        if p.name not in kf_names and t0 <= float(p.stem) <= t1
    ][:n_frames]

    frames = []
    for name in non_kf:
        img = cv2.imread(str(rgb_dir / name))
        if img is None:
            raise FileNotFoundError(f"이미지 없음: {rgb_dir / name}")
        img = cv2.resize(img, (W, H))
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        frames.append((name, rgb, None))   # GT 없음

    print(f"[load] non-KF {len(frames)} 프레임 로드 완료  "
          f"({t0}~{t1}s, 해상도 {W}×{H})")
    return frames


# ── 집계 ─────────────────────────────────────────────────────────────────────
def aggregate(records: list, has_gt: bool = True) -> dict:
    """list of dict(ms, state, [trans_cm, rot_deg,] [n_inliers]) → 집계 dict.

    has_gt=False: non-KF 모드 — trans_cm/rot_deg 없음.
    """
    ms_arr = np.array([r["ms"]        for r in records])
    ok_arr = np.array([r["state"] == OK for r in records])

    n    = len(records)
    n_ok = int(ok_arr.sum())
    fps_arr = 1000.0 / np.where(ms_arr > 0, ms_arr, np.nan)

    result = dict(
        n_frames      = n,
        ms_median     = float(np.median(ms_arr)),
        ms_max        = float(np.max(ms_arr)),
        fps_median    = float(np.nanmedian(fps_arr)),
        fps_30_target = bool(float(np.median(ms_arr)) <= (1000.0 / FPS_TARGET)),
        reloc_ok_rate = float(n_ok / n),
        fps_per_frame = [float(v) for v in fps_arr],
    )

    # inlier 수 (PnPLocalizer / MockPnP 모두 기록)
    if records and "n_inliers" in records[0]:
        inl_arr = np.array([r["n_inliers"] for r in records])
        result.update(dict(
            inliers_median    = float(np.median(inl_arr)),
            inliers_min       = float(np.min(inl_arr)),
            inliers_max       = float(np.max(inl_arr)),
            inliers_per_frame = [int(v) for v in inl_arr],
        ))

    if has_gt:
        dt_arr = np.array([r["trans_cm"] for r in records])
        dr_arr = np.array([r["rot_deg"]  for r in records])
        tight_ok = (dt_arr <= THRESH_TRANS_CM) & (dr_arr < THRESH_ROT_DEG)
        result.update(dict(
            trans_cm_median = float(np.median(dt_arr)),
            trans_cm_max    = float(np.max(dt_arr)),
            rot_deg_median  = float(np.median(dr_arr)),
            rot_deg_max     = float(np.max(dr_arr)),
            tight_ok_frac   = float(tight_ok.sum()) / n,
            thresh_trans_cm = THRESH_TRANS_CM,
            thresh_rot_deg  = THRESH_ROT_DEG,
            trans_per_frame = [float(v) for v in dt_arr],
            rot_per_frame   = [float(v) for v in dr_arr],
        ))

    return result


# ── 출력 ─────────────────────────────────────────────────────────────────────
def print_table(agg: dict, localizer_name: str, has_gt: bool = True):
    ok_mark = "OK" if agg["fps_30_target"] else "X"
    mode    = "KF (upper-bound)" if has_gt else "non-KF (real query)"
    print()
    print(f"{'='*58}")
    print(f"  PnP Localizer Benchmark  [{localizer_name}]  {mode}")
    print(f"{'='*58}")
    print(f"  프레임 수             : {agg['n_frames']}")
    print(f"  속도  median          : {agg['ms_median']:.1f} ms  ({agg['fps_median']:.1f} FPS)")
    print(f"  속도  max             : {agg['ms_max']:.1f} ms")
    print(f"  30 FPS (33ms) 달성    : [{ok_mark}]")
    if has_gt:
        print(f"  포즈오차 trans cm     : median {agg['trans_cm_median']:.2f}  max {agg['trans_cm_max']:.2f}")
        print(f"  포즈오차 rot deg      : median {agg['rot_deg_median']:.2f}  max {agg['rot_deg_max']:.2f}")
        print(f"  §2 허용오차 달성율    : {agg['tight_ok_frac']*100:.1f}%"
              f"  (<{THRESH_ROT_DEG}° AND ≤{THRESH_TRANS_CM}cm)")
    if "inliers_median" in agg:
        print(f"  inlier 수  median     : {agg['inliers_median']:.0f}"
              f"  min {agg['inliers_min']:.0f}  max {agg['inliers_max']:.0f}"
              f"  (OK 기준: >={INLIER_OK})")
    print(f"  전역 reloc OK율       : {agg['reloc_ok_rate']*100:.1f}%")
    print(f"{'='*58}")


def save_json(agg: dict, localizer_name: str, path: Path):
    out = dict(localizer=localizer_name, **agg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[json] {path}")


def save_plot(agg: dict, localizer_name: str, path: Path, has_gt: bool = True):
    fps_arr = np.array(agg["fps_per_frame"])
    fig, ax = plt.subplots(figsize=(7, 5))

    if has_gt:
        dt_arr = np.array(agg["trans_per_frame"])
        ax.scatter(fps_arr, dt_arr, s=40, alpha=0.75, color="steelblue", zorder=3)
        ax.axhline(THRESH_TRANS_CM, color="orange", linestyle="--", linewidth=1.2,
                   label=f"{THRESH_TRANS_CM:.0f} cm ref")
        ax.set_ylabel("Pose error (cm, camera center)")
        subtitle = "KF upper-bound"
    elif "inliers_per_frame" in agg:
        inl_arr = np.array(agg["inliers_per_frame"])
        ax.scatter(fps_arr, inl_arr, s=40, alpha=0.75, color="darkorange", zorder=3)
        ax.axhline(INLIER_OK, color="red", linestyle="--", linewidth=1.2,
                   label=f"inlier OK>={INLIER_OK}")
        ax.set_ylabel("PnP inliers")
        subtitle = "non-KF real query"
    else:
        ax.scatter(fps_arr, np.zeros_like(fps_arr), s=40, alpha=0.75, color="gray", zorder=3)
        ax.set_ylabel("(no GT)")
        subtitle = "non-KF"

    ax.axvline(FPS_TARGET, color="blue", linestyle="--", linewidth=1.2,
               label=f"{FPS_TARGET:.0f} FPS ref")
    ax.set_xlabel("FPS (1000 / ms/frame)")
    ax.set_title(f"PnP Benchmark [{localizer_name}] {subtitle}  (n={agg['n_frames']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ── Localizer 생성 ───────────────────────────────────────────────────────────
def build_localizer(name: str, gt_frames: list, K_mat: np.ndarray) -> Localizer:
    if name == "mock":
        # GT 없는 non-KF 프레임(vm=None)엔 단위행렬 대체
        gt_poses = [
            (f, vm if vm is not None else np.eye(4, dtype=np.float32))
            for f, _, vm in gt_frames
        ]
        return MockPnP(gt_poses=gt_poses)

    if name == "pnp":
        pnp_path = ROOT / "scripts" / "pnp_localizer.py"
        if not pnp_path.exists():
            raise FileNotFoundError(
                f"pnp_localizer.py 없음: {pnp_path}\n"
                "  --localizer mock 으로 하니스만 실행 가능합니다."
            )
        if not FEAT.exists():
            raise FileNotFoundError(
                f"feature_map.npz 없음: {FEAT}\n"
                "  feature map 생성 후 재시도하세요."
            )
        spec = importlib.util.spec_from_file_location("pnp_localizer", pnp_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.PnPLocalizer(feature_map_path=FEAT, K=K_mat)

    raise ValueError(f"알 수 없는 localizer: {name!r} (pnp|mock)")


# ── 벤치마크 본체 ─────────────────────────────────────────────────────────────
def run_benchmark(localizer_name: str, n_frames: int, frames_mode: str = "kf") -> dict:
    """
    frames_mode: "kf"     — images.txt KF 프레임 (GT 포즈 있음, 정확도 상한)
                 "non-kf" — 맵에 없는 non-KF 프레임 (GT 없음, 실전 측정)
    """
    has_gt = (frames_mode == "kf")

    # GT 포즈·카메라 로드
    W, H, fx, fy, cx, cy = read_colmap_cameras(SP / "cameras.txt")
    K_mat  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    images = read_colmap_images(SP / "images.txt")

    # 프레임 사전 로드 (타이밍 루프 전에 전부 메모리로)
    if has_gt:
        frames = load_frames(PROC, images, n_frames, W, H)
    else:
        frames = load_non_kf_frames(PROC, images, n_frames, W, H)

    # Localizer 초기화
    localizer = build_localizer(localizer_name, frames, K_mat)

    # warmup — 첫 호출에 feature matcher/KD-tree 초기화가 끼지 않도록
    _dummy = np.zeros((H, W, 3), dtype=np.float32)
    localizer.relocalize(_dummy, hint=None)
    # mock의 경우 warmup 소진된 인덱스 초기화
    if hasattr(localizer, "_idx"):
        localizer._idx = 0

    # 타이밍 루프 (relocalize only, hint=None → 전역 reloc)
    records = []
    mode_label = "KF (GT 있음)" if has_gt else "non-KF (GT 없음)"
    print(f"\n[bench] {localizer_name} | {mode_label} | {len(frames)} frames | hint=None")
    if has_gt:
        print(f"  {'프레임':<22}  {'ms':>7}  {'trans cm':>8}  {'rot °':>6}  {'inliers':>7}  state")
        print(f"  {'-'*22}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*7}  -----")
    else:
        print(f"  {'프레임':<22}  {'ms':>7}  {'inliers':>7}  state")
        print(f"  {'-'*22}  {'-'*7}  {'-'*7}  -----")

    for i, (name, rgb, gt_vm) in enumerate(frames):
        t0  = time.perf_counter()
        res = localizer.relocalize(rgb, hint=None)
        ms  = (time.perf_counter() - t0) * 1000.0
        n_inl = getattr(localizer, "last_n_inliers", -1)

        rec = dict(frame=name, ms=ms, state=res.state,
                   conf=res.confidence, n_inliers=n_inl)
        if has_gt:
            dt_cm, dr_deg = pose_err(res.T_map_cam, gt_vm)
            rec["trans_cm"] = dt_cm
            rec["rot_deg"]  = dr_deg
            print(f"  [{i+1:3d}/{len(frames)}] {name:<18}  {ms:7.1f}  "
                  f"{dt_cm:8.2f}  {dr_deg:6.2f}  {n_inl:7d}  {res.state}")
        else:
            print(f"  [{i+1:3d}/{len(frames)}] {name:<18}  {ms:7.1f}  "
                  f"{n_inl:7d}  {res.state}")
        records.append(rec)

    # 집계
    agg = aggregate(records, has_gt=has_gt)
    print_table(agg, localizer_name, has_gt=has_gt)

    # JSON (모드별 분리)
    suffix  = "" if has_gt else "_nonkf"
    json_path = OUT_DIR / f"pnp_benchmark{suffix}.json"
    png_path  = ROOT / f"docs/assets/report/pnp_benchmark{suffix}.png"
    save_json(agg, localizer_name, json_path)
    save_plot(agg, localizer_name, png_path, has_gt=has_gt)

    return agg


def main():
    parser = argparse.ArgumentParser(description="PnP Localizer 벤치마크")
    parser.add_argument(
        "--localizer", choices=["pnp", "mock"], default="mock",
        help="pnp: 실제 PnPLocalizer, mock: MockPnP (기본)",
    )
    parser.add_argument(
        "--frames", choices=["kf", "non-kf"], default="kf",
        help="kf: COLMAP KF (GT 포즈 있음, 상한), non-kf: 맵 밖 프레임 (실전) (기본 kf)",
    )
    parser.add_argument(
        "--n-frames", type=int, default=N_FRAMES_DEFAULT,
        help=f"벤치마크 프레임 수 (기본 {N_FRAMES_DEFAULT})",
    )
    args = parser.parse_args()
    run_benchmark(args.localizer, args.n_frames, args.frames)


if __name__ == "__main__":
    main()
