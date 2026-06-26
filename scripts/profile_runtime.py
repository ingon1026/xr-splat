#!/usr/bin/env python3
"""profile_runtime.py — 런타임 photometric 트래킹의 속도/품질 프로파일 (실시간화).

레버: 렌더 해상도(scale)·track iters·가우시안 자산(full 2M vs lite 643k).
각 조합으로 in-region 연속 KF를 트래킹하며 프레임당 localize 지연(ms)·FPS·트래킹 품질(%OK, median conf) 측정.
→ "실시간(30FPS=33ms)까지 얼마나 가는가 / 속도-품질 트레이드오프" 정량화.

usage: python scripts/profile_runtime.py [--frames 12]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from pipeline.runtime import OK  # noqa: E402
from runtime_localizer import PhotometricLocalizer  # noqa: E402

DEV = "cuda"
SCENE = "ros2_bag2_home_rgbd_orbframe"


def load_stream(n):
    proc = ROOT / "data/processed" / SCENE
    sp = proc / "colmap/sparse/0"
    imgs = read_colmap_images(sp / "images.txt")
    names = sorted([k for k in imgs], key=lambda x: float(Path(x).stem))[:n]
    W0, H0, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    frames = []
    for nm in names:
        qw, qx, qy, qz, tx, ty, tz = imgs[nm]
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix(); vm[:3, 3] = [tx, ty, tz]
        rgb = cv2.imread(str(proc / "rgb" / nm))[:, :, ::-1].astype(np.float32) / 255.0
        frames.append((rgb, vm))
    return frames, (W0, H0, fx, fy, cx, cy)


def run_config(frames, intr, ply, scale, track_iters):
    W0, H0, fx, fy, cx, cy = intr
    W, H = int(W0 * scale), int(H0 * scale)
    K = torch.tensor([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1.0]], device=DEV)
    loc = PhotometricLocalizer(ROOT / ply, K, W, H, reloc_iters=150, track_iters=track_iters)
    rs = [(cv2.resize(rgb, (W, H)), vm) for rgb, vm in frames]
    # frame0: seed(5cm 섭동)에서 relocalize
    from pipeline.runtime import PoseResult, LOST
    hint = rs[0][1].copy(); hint[:3, 3] += np.array([0.05, 0, 0], np.float32)
    prior = loc.relocalize(rs[0][0], hint)
    lat, confs, oks = [], [], []
    for rgb, _ in rs[1:]:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        prior = loc.track(rgb, prior)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000); confs.append(prior.confidence); oks.append(prior.state == OK)
    del loc; torch.cuda.empty_cache()
    ms = float(np.median(lat))
    return dict(ply=Path(ply).name, scale=scale, track_iters=track_iters, W=W, H=H,
                ms_per_frame=round(ms, 1), fps=round(1000 / ms, 1),
                pct_ok=round(100 * np.mean(oks), 0), conf_median=round(float(np.median(confs)), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()
    frames, intr = load_stream(args.frames)
    print(f"[prof] {len(frames)} frames, 워밍업 포함 frame0=relocalize", flush=True)

    FULL = "outputs/ros2_bag2_home_rgbd_orbframe/gsplat_mcmc2m/scene.ply"
    LITE = "outputs/ros2_bag2_home_rgbd_orbframe/gsplat_mcmc2m/scene_lite.ply"
    configs = [
        (FULL, 1.0, 50),   # 베이스라인(현재 런타임 데모 설정)
        (FULL, 0.5, 50), (FULL, 0.5, 20), (FULL, 0.25, 20), (FULL, 0.25, 10),
        (LITE, 0.5, 20), (LITE, 0.25, 20), (LITE, 0.25, 10),
    ]
    rows = []
    for ply, sc, it in configs:
        r = run_config(frames, intr, ply, sc, it)
        rows.append(r)
        print(f"  {r['ply']:<16} scale={r['scale']} iters={r['track_iters']:>2} "
              f"{r['W']}x{r['H']:<4} → {r['ms_per_frame']:>7}ms  {r['fps']:>5}FPS  "
              f"OK {r['pct_ok']:>3.0f}%  conf {r['conf_median']}", flush=True)

    out = ROOT / "outputs/ros2_bag2_home_rgbd_orbframe/gsplat_mcmc2m/runtime_profile.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    base = rows[0]
    print(f"\n[prof] 베이스라인 {base['fps']}FPS({base['ms_per_frame']}ms) → "
          f"최적 {max(r['fps'] for r in rows if r['pct_ok']>=90)}FPS(OK≥90% 유지). 실시간 30FPS=33ms 기준 거리.", flush=True)
    print(f"[prof] json → {out}\n[prof] DONE", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in rows:
            c = "tab:green" if r["pct_ok"] >= 90 else "tab:red"
            ax.scatter(r["fps"], r["conf_median"], s=80, c=c)
            ax.annotate(f"{r['ply'][:4]} s{r['scale']} i{r['track_iters']}", (r["fps"], r["conf_median"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axvline(30, ls="--", c="gray", label="real-time 30 FPS")
        ax.set_xlabel("FPS (localize only)"); ax.set_ylabel("tracking confidence (median)")
        ax.set_title("Runtime tracking: speed vs quality (green=OK≥90%)"); ax.legend(); ax.grid(True)
        fig.tight_layout(); op = ROOT / "docs/assets/report/runtime_profile.png"; fig.savefig(op, dpi=110)
        print(f"[prof] plot → {op}", flush=True)
    except Exception as e:
        print(f"[prof] plot skip: {e}", flush=True)


if __name__ == "__main__":
    main()
