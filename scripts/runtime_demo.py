#!/usr/bin/env python3
"""runtime_demo.py — run_loop 런타임 루프 데모.

기준 COLMAP KF 포즈의 앞쪽 ~40개를 스트림으로 흘려 위치추정→렌더 루프를 돌린다.

출력:
  docs/assets/report/runtime_loop.mp4        — 실제|렌더 side-by-side + 상태 오버레이
  docs/assets/report/runtime_loop_state.png  — 프레임별 confidence 그래프 (OK/LOST 색)
  docs/assets/report/runtime_loop_log.json   — 프레임별 pose 요약·state·confidence

사용법:
  python scripts/runtime_demo.py [--localizer mock|photometric] [--frames 40] [--conf-thresh 0.5]

--localizer mock: MockLocalizer (CPU, GPU 불필요, 루프·출력 개발용)
--localizer photometric: PhotometricLocalizer (scripts/runtime_localizer.py 필요, GPU)

통합 지점:
  PhotometricLocalizer는 scripts/runtime_localizer.py 에 구현 예정.
  생성자에서 초기 추정 포즈(섭동된 gt0)를 받아, hint=None일 때 fallback으로 사용해야 한다.
  (run_loop은 첫 프레임을 relocalize(rgb, hint=None)으로 호출하므로 Localizer 내부에서 처리.)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.runtime import Localizer, PoseResult, OK, LOST, run_loop  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from pipeline.gsplat_io import load_ply, render as gs_render  # noqa: E402

# ---------------------------------------------------------------------------
# 기본 경로
# ---------------------------------------------------------------------------

PLY_PATH   = ROOT / "outputs/ros2_bag2_home_rgbd_orbframe/gsplat_mcmc2m/scene.ply"
COLMAP_DIR = ROOT / "data/processed/ros2_bag2_home_rgbd_orbframe/colmap/sparse/0"
RGB_DIR    = ROOT / "data/processed/ros2_bag2_home_rgbd_orbframe/rgb"
OUT_DIR    = ROOT / "docs/assets/report"

# ---------------------------------------------------------------------------
# COLMAP 로더
# ---------------------------------------------------------------------------

def colmap_to_viewmat(qw, qx, qy, qz, tx, ty, tz) -> np.ndarray:
    """COLMAP Tcw 파라미터 → 4x4 viewmat (Tcw, world→cam)."""
    R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cw
    T[:3, 3]  = [tx, ty, tz]
    return T


def load_kf_stream(colmap_dir: Path, rgb_dir: Path, n_frames: int):
    """이미지 파일이 존재하는 KF를 IMAGE_ID 순으로 n_frames개 반환.

    반환: list of (name, rgb[H,W,3] float32 0~1, viewmat[4,4])
    """
    imgs = read_colmap_images(str(colmap_dir / "images.txt"))
    # read_colmap_images: {NAME: (qw,qx,qy,qz,tx,ty,tz)}
    # IMAGE_ID 순서가 필요하므로 images.txt를 직접 파싱해 ID 포함
    id_name = {}
    data = [l for l in open(colmap_dir / "images.txt") if not l.startswith("#") and l.strip()]
    for line in data[0::2]:
        s = line.split()
        if len(s) >= 10:
            id_name[int(s[0])] = s[9]

    sorted_ids = sorted(id_name.keys())
    stream = []
    for iid in sorted_ids:
        name = id_name[iid]
        rgb_path = rgb_dir / name
        if not rgb_path.exists():
            continue
        q = imgs[name]  # (qw,qx,qy,qz,tx,ty,tz)
        viewmat = colmap_to_viewmat(*q)
        img = cv2.imread(str(rgb_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        stream.append((name, rgb, viewmat))
        if len(stream) >= n_frames:
            break

    if len(stream) == 0:
        raise RuntimeError(f"RGB 프레임을 찾을 수 없음: {rgb_dir}")
    return stream


# ---------------------------------------------------------------------------
# MockLocalizer
# ---------------------------------------------------------------------------

class MockLocalizer(Localizer):
    """결정론적 MockLocalizer — gt_poses[i]+노이즈 반환, confidence 스케줄로 LOST 시뮬레이션.

    call 경로(relocalize/track)는 self.calls에 기록.
    frame-0 hint는 run_loop이 None으로 넘기므로 내부에서 gt[0]+섭동으로 처리.
    """

    def __init__(self, gt_poses: list, conf_schedule: list = None, rng_seed: int = 42):
        """
        gt_poses: 프레임별 4x4 Tcw (ground-truth viewmat).
        conf_schedule: 프레임별 confidence 값 목록 (None이면 기본 스케줄 사용).
        """
        self._gt    = gt_poses
        self._conf  = conf_schedule if conf_schedule is not None else self._default_schedule(len(gt_poses))
        self._rng   = np.random.default_rng(rng_seed)
        self._idx   = 0
        self.calls  = []  # [("relocalize"|"track", frame_idx)]

    @staticmethod
    def _default_schedule(n: int) -> list:
        """데모용 기본 confidence 스케줄:
        0~24: 0.75 (OK, track 경로)
        25~27: 0.3  (conf<thresh → 다음 프레임 relocalize 경로 트리거)
        28   : 0.05 (LOST)
        29+  : 0.7  (회복)
        """
        conf = []
        for i in range(n):
            if i < 25:
                conf.append(0.75)
            elif i < 28:
                conf.append(0.30)
            elif i == 28:
                conf.append(0.05)
            else:
                conf.append(0.70)
        return conf

    def _next_result(self, method: str) -> PoseResult:
        i = self._idx
        self._idx += 1
        self.calls.append((method, i))

        gt = self._gt[i].copy()
        # 소형 노이즈: 1cm + 0.3°
        noise_t = self._rng.normal(0, 0.01, 3)
        noise_r = self._rng.normal(0, np.deg2rad(0.3), 3)
        R_noise  = Rot.from_rotvec(noise_r).as_matrix()
        gt[:3, :3] = R_noise @ gt[:3, :3]
        gt[:3,  3] = gt[:3, 3] + noise_t

        conf = float(self._conf[i]) if i < len(self._conf) else 0.7
        state = OK if conf > 0.15 else LOST
        return PoseResult(T_map_cam=gt, state=state, confidence=conf)

    def relocalize(self, rgb: np.ndarray, hint=None) -> PoseResult:
        return self._next_result("relocalize")

    def track(self, rgb: np.ndarray, prior: PoseResult) -> PoseResult:
        return self._next_result("track")


# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------

def _resize(img_f32, scale: float = 0.5):
    """float32 [H,W,3] → uint8 BGR, 스케일 적용."""
    h, w = img_f32.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    bgr = cv2.cvtColor((img_f32 * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (nw, nh))


def _overlay_text(frame_bgr, lines: list, org=(10, 25), dy=22,
                  font=cv2.FONT_HERSHEY_SIMPLEX, fs=0.6, thick=1):
    """프레임에 텍스트 줄 오버레이 (그림자 효과)."""
    x, y = org
    for line in lines:
        color = (0, 200, 0) if "OK" in line else (0, 0, 220) if "LOST" in line else (255, 255, 255)
        cv2.putText(frame_bgr, line, (x+1, y+1), font, fs, (0,0,0), thick+1, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (x,   y  ), font, fs, color,   thick,   cv2.LINE_AA)
        y += dy
    return frame_bgr


def write_mp4(results, stream, out_path: Path, scale: float = 0.5):
    """side-by-side mp4 저장 (실제 | 렌더)."""
    # 프레임 크기 추론
    _, rgb0, _ = stream[0]
    h, w = rgb0.shape[:2]
    nh, nw = int(h * scale), int(w * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, 10.0, (nw * 2, nh))

    for r in results:
        i   = r["idx"]
        _, rgb_real, _ = stream[i]
        state = r["pose"].state
        conf  = r["pose"].confidence

        left  = _resize(rgb_real, scale)

        if r["render"] is not None:
            right = _resize(r["render"], scale)
        else:
            # LOST: 검정 프레임
            right = np.zeros_like(left)

        # 라벨 오버레이
        label_lines = [f"F{i:03d}", f"state:{state}", f"conf:{conf:.2f}"]
        _overlay_text(left,  label_lines, org=(8, 22))
        _overlay_text(right, ["(render)" if r["render"] is not None else "(LOST)"],
                      org=(8, 22))

        frame = np.hstack([left, right])
        vw.write(frame)

    vw.release()
    print(f"[demo] mp4 저장: {out_path} ({len(results)} frames)")


def write_state_graph(results, out_path: Path):
    """confidence 그래프 (OK=녹색, LOST=빨강)."""
    idxs  = [r["idx"]               for r in results]
    confs = [r["pose"].confidence    for r in results]
    states= [r["pose"].state         for r in results]

    fig, ax = plt.subplots(figsize=(10, 3))
    for i, (c, s) in enumerate(zip(confs, states)):
        color = "green" if s == OK else "red"
        ax.bar(i, c, color=color, alpha=0.7, width=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="conf_thresh=0.5")
    ax.set_xlabel("frame")
    ax.set_ylabel("confidence")
    ax.set_title("Runtime Loop — Confidence & State (green=OK, red=LOST)")
    ax.legend()
    ax.set_xlim(-0.5, len(results) - 0.5)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"[demo] 상태 그래프: {out_path}")


def write_json_log(results, out_path: Path):
    """프레임별 요약 JSON."""
    log = []
    for r in results:
        T = r["pose"].T_map_cam
        log.append(dict(
            idx=r["idx"],
            state=r["pose"].state,
            confidence=round(float(r["pose"].confidence), 4),
            t_cam=T[:3, 3].tolist(),
        ))
    out_path.write_text(json.dumps(log, indent=2))
    print(f"[demo] JSON 로그: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_render_fn(ply_path: Path, W: int, H: int, fx: float, fy: float, cx: float, cy: float):
    """GPU가 필요한 gaussian render_fn 생성. PLY 로드 포함."""
    import torch
    device = "cuda"
    g = load_ply(str(ply_path), device=device)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)

    def render_fn(T_map_cam: np.ndarray):
        vm = torch.tensor(T_map_cam, device=device, dtype=torch.float32)
        with torch.no_grad():
            rgb = gs_render(g, vm, K, W, H)
        return rgb.cpu().numpy()

    return render_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--localizer", choices=["mock", "photometric"], default="mock")
    ap.add_argument("--frames",    type=int, default=40, help="KF 스트림 길이")
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--scale",     type=float, default=0.5, help="mp4 출력 스케일")
    ap.add_argument("--ply",       type=Path, default=PLY_PATH)
    ap.add_argument("--colmap",    type=Path, default=COLMAP_DIR)
    ap.add_argument("--rgb-dir",   type=Path, default=RGB_DIR)
    ap.add_argument("--out-dir",   type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 데이터 로드 ---
    print(f"[demo] KF 스트림 로드 ({args.frames}개)…")
    stream = load_kf_stream(args.colmap, args.rgb_dir, args.frames)
    print(f"[demo] 실제 로드: {len(stream)}개 KF")

    W, H, fx, fy, cx, cy = read_colmap_cameras(str(args.colmap / "cameras.txt"))

    # --- 렌더 함수 ---
    print(f"[demo] Gaussian PLY 로드: {args.ply}")
    render_fn = build_render_fn(args.ply, W, H, fx, fy, cx, cy)

    # --- Localizer 선택 ---
    gt_poses = [vm for (_, _, vm) in stream]

    if args.localizer == "mock":
        localizer = MockLocalizer(gt_poses)
        print(f"[demo] MockLocalizer 사용 ({len(gt_poses)}개 gt 포즈)")

    elif args.localizer == "photometric":
        import importlib.util
        import torch
        loc_path = ROOT / "scripts" / "runtime_localizer.py"
        spec = importlib.util.spec_from_file_location("runtime_localizer", loc_path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device="cuda")
        inner = mod.PhotometricLocalizer(ply_path=args.ply, K=K, W=W, H=H)   # 실제 생성자 시그니처

        # frame-0 hint 주입 어댑터: run_loop이 frame-0에서 relocalize(rgb, hint=None)을 부르는데
        # PhotometricLocalizer는 hint=None이면 LOST(전역 place-recognition 범위 밖). 실시스템에선
        # 거친 localizer(COLMAP/ORB)가 init을 주듯, 여기선 gt0를 5cm 섭동한 hint로 seed한다.
        class SeededLocalizer(Localizer):
            def __init__(self, inner, hint0):
                self.inner, self.hint0, self.used = inner, hint0, False
            def relocalize(self, rgb, hint=None):
                if hint is None and not self.used:
                    self.used = True; hint = self.hint0
                return self.inner.relocalize(rgb, hint)
            def track(self, rgb, prior):
                return self.inner.track(rgb, prior)

        hint0 = gt_poses[0].copy(); hint0[:3, 3] += np.array([0.05, 0.0, 0.0], np.float32)
        localizer = SeededLocalizer(inner, hint0)
        print("[demo] PhotometricLocalizer 사용 (frame-0 hint=gt0+5cm seed)")
    else:
        raise ValueError(f"알 수 없는 localizer: {args.localizer}")

    # --- 루프 실행 ---
    rgb_frames = [rgb for (_, rgb, _) in stream]
    print(f"[demo] run_loop 시작 (conf_thresh={args.conf_thresh})…")
    results = run_loop(localizer, rgb_frames, render_fn, conf_thresh=args.conf_thresh)

    # 상태 통계
    ok_cnt   = sum(1 for r in results if r["pose"].state == OK)
    lost_cnt = len(results) - ok_cnt
    print(f"[demo] 완료: {ok_cnt} OK / {lost_cnt} LOST / {len(results)} total")

    if hasattr(localizer, "calls"):
        n_rel = sum(1 for m, _ in localizer.calls if m == "relocalize")
        n_trk = sum(1 for m, _ in localizer.calls if m == "track")
        print(f"[demo] relocalize={n_rel}, track={n_trk}")

    # --- 출력 저장 ---
    write_mp4(results, stream, args.out_dir / "runtime_loop.mp4", scale=args.scale)
    write_state_graph(results, args.out_dir / "runtime_loop_state.png")
    write_json_log(results, args.out_dir / "runtime_loop_log.json")


if __name__ == "__main__":
    main()
