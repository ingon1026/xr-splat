#!/usr/bin/env python3
"""validate_pnp_render.py — feature-PnP localizer 정직 검증 (non-KF + 풀자산 render-vs-real).

맵(feature_map)에 없는 non-KF query를 PnP로 전역 localize → **풀 자산 scene.ply(2M)** 로 렌더 →
실제 프레임과 PSNR. GT 포즈 불필요(이미지가 정답). 렌더는 항상 풀 자산(품질 제약 준수).
PnP는 CPU(localize), 렌더는 GPU(검증용). usage: python scripts/validate_pnp_render.py
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from pipeline.gsplat_io import load_ply, render  # noqa: E402
from pipeline.runtime import OK  # noqa: E402
from pnp_localizer import PnPLocalizer  # noqa: E402

DEV = "cuda"
SC = "ros2_bag2_home_rgbd_orbframe"


def main():
    proc = ROOT / "data/processed" / SC
    sp = proc / "colmap/sparse/0"
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float32)
    kf = set(read_colmap_images(sp / "images.txt"))
    allrgb = [l.split()[1].split('/')[-1] for l in open(proc / "rgb.txt") if l.strip() and not l.startswith('#')]
    nonkf = sorted([n for n in allrgb if n not in kf and 68.0 <= float(n[:-4]) <= 85.6],
                   key=lambda x: float(x[:-4]))[:30]
    print(f"[honest] non-KF query {len(nonkf)}개 (맵에 없음)", flush=True)

    loc = PnPLocalizer(ROOT / "outputs" / SC / "feature_map.npz", K_np)
    g = load_ply(ROOT / "outputs" / SC / "gsplat_mcmc2m/scene.ply", DEV)   # 풀 자산(2M) — 품질 제약
    Kt = torch.tensor(K_np, device=DEV)
    rows = []
    for n in nonkf:
        rgb = cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0
        t0 = time.perf_counter(); pr = loc.relocalize(rgb); ms = (time.perf_counter() - t0) * 1000
        psnr = None
        if pr.state == OK:
            with torch.no_grad():
                vm = torch.tensor(np.asarray(pr.T_map_cam, np.float32), device=DEV)
                im = render(g, vm, Kt, W, H)
                gt = torch.tensor(rgb.copy(), device=DEV)
                psnr = float(-10 * np.log10(max(torch.mean((im.clamp(0, 1) - gt) ** 2).item(), 1e-10)))
        rows.append(dict(name=n, state=pr.state, conf=round(pr.confidence, 3),
                         ms=round(ms, 1), psnr=round(psnr, 2) if psnr else None))
        print(f"  {n}: {pr.state} conf{pr.confidence:.2f} {ms:.1f}ms PSNR {None if psnr is None else round(psnr,1)}", flush=True)

    okr = [r for r in rows if r['state'] == OK]
    ms = np.array([r['ms'] for r in rows])
    print(f"\n[honest] non-KF {len(rows)}개 | 전역성공 {len(okr)}/{len(rows)} ({100*len(okr)/len(rows):.0f}%)", flush=True)
    print(f"  속도: median {np.median(ms):.1f}ms = {1000/np.median(ms):.1f}FPS", flush=True)
    if okr:
        ps = np.array([r['psnr'] for r in okr]); cf = np.array([r['conf'] for r in okr])
        print(f"  render-vs-real PSNR(풀자산): median {np.median(ps):.2f}  (포즈 정확도 GT-free 지표)", flush=True)
        print(f"  conf median {np.median(cf):.2f}", flush=True)
    (ROOT / "outputs" / SC / "pnp_render_validation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("[honest] DONE", flush=True)


if __name__ == "__main__":
    main()
