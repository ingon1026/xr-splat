#!/usr/bin/env python3
"""view_gsplat.py — 학습된 가우시안 자산을 로컬 인터랙티브 뷰어로 띄운다(자유 시점).

gsplat 자체 뷰어(nerfview+viser): 렌더는 서버(우리 CUDA)에서 진짜 gsplat rasterizer로
수행되어 브라우저로 스트림됨 → WSL OpenGL 이슈 우회, 외부 업로드 0(파일 기기 밖으로 안 나감),
학습과 동일 rasterizer라 충실. 학습 궤적 밖 시점도 자유 회전/이동으로 검증 가능.

usage: python view_gsplat.py --ply <scene.ply> [--port 8080]
       → http://localhost:<port> 를 브라우저에서 열기
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "gsplat" / "examples"))  # GsplatViewer
import viser  # noqa: E402
from nerfview import CameraState, RenderTabState  # noqa: E402
from gsplat_viewer import GsplatViewer  # noqa: E402
from pipeline.gsplat_io import load_ply, render  # noqa: E402

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    g = load_ply(args.ply, DEV)
    n = g["means"].shape[0]
    print(f"[view] loaded {args.ply} ({n:,} gaussians, sh_degree={g['sh_degree']})", flush=True)

    @torch.no_grad()
    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        if getattr(render_tab_state, "preview_render", False):
            w, h = render_tab_state.render_width, render_tab_state.render_height
        else:
            w, h = render_tab_state.viewer_width, render_tab_state.viewer_height
        c2w = torch.from_numpy(camera_state.c2w).float().to(DEV)
        K = torch.from_numpy(camera_state.get_K((w, h))).float().to(DEV)
        viewmat = torch.linalg.inv(c2w)                     # c2w → Tcw(world→cam)
        img = render(g, viewmat, K, w, h)                   # [H,W,3] float 0-1, 학습과 동일 rasterizer
        render_tab_state.total_gs_count = n
        render_tab_state.rendered_gs_count = n
        return img.clamp(0, 1).cpu().numpy()

    server = viser.ViserServer(port=args.port, verbose=False)
    GsplatViewer(server=server, render_fn=render_fn, output_dir=ROOT / "outputs" / "_viewer", mode="rendering")
    print(f"[view] READY → http://localhost:{args.port}  (Ctrl+C 종료)", flush=True)
    while True:
        time.sleep(2)


if __name__ == "__main__":
    main()
