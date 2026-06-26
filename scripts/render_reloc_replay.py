#!/usr/bin/env python3
"""render_reloc_replay.py — reloc pose로 orbframe 가우시안 렌더 + 실제 rgb와 side-by-side 영상.
usage: render_reloc_replay.py --reloc <CameraTrajectory_reloc.txt> --scene ros2_bag2_home_rgbd_orbframe
       --rgb-scene ros2_bag2_home_rgbd [--start 0 --end -1 --stride 2 --width 640 --fps 18]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, cv2, imageio.v2 as imageio
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.backproject import read_colmap_cameras
DEV = "cuda"; ROOT = Path(__file__).resolve().parents[1]


def twc_line_to_viewmat(tx, ty, tz, qx, qy, qz, qw):
    R_wc = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    R_cw = R_wc.T
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R_cw; vm[:3, 3] = -R_cw @ np.array([tx, ty, tz])
    return vm


def main():
    # Keep gsplat import lazy so pure pose tests can run without CUDA/gsplat installed.
    from pipeline.gsplat_io import load_ply, render

    ap = argparse.ArgumentParser()
    ap.add_argument("--reloc", required=True, type=Path)
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--rgb-scene", default="ros2_bag2_home_rgbd")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--gs-tag", default="", help="gsplat 태그: 비면 gsplat/scene.ply, 있으면 gsplat_<tag>/scene.ply")
    ap.add_argument("--out", type=Path, default=ROOT/"docs/assets/reloc_replay_home.mp4")
    a = ap.parse_args()
    proc = ROOT/"data/processed"/a.scene; rgbdir = ROOT/"data/processed"/a.rgb_scene/"rgb"
    W, H, fx, fy, cx, cy = read_colmap_cameras(proc/"colmap/sparse/0/cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], device=DEV)
    gs_sub = "gsplat" + (f"_{a.gs_tag}" if a.gs_tag else "")
    g = load_ply(ROOT/"outputs"/a.scene/gs_sub/"scene.ply", DEV)
    lines = [l.split() for l in open(a.reloc) if len(l.split()) >= 9]
    end = len(lines) if a.end < 0 else a.end
    lines = lines[a.start:end:a.stride]
    frames, ok = [], 0
    for s in lines:
        ts = s[0]; vals = list(map(float, s[1:8])); state = int(float(s[8]))
        rgb = cv2.imread(str(rgbdir/f"{ts}.png"))            # 실제 프레임
        if rgb is None: continue
        rgb = rgb[:, :, ::-1]
        if state == 2:                                       # reloc/track OK → 렌더
            vm = torch.tensor(twc_line_to_viewmat(*vals), device=DEV)
            with torch.no_grad():
                rnd = (render(g, vm, K, W, H).detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)
            ok += 1
        else:                                                # LOST → 회색 + 텍스트
            rnd = np.full_like(np.asarray(rgb), 40)
            cv2.putText(rnd, "RELOC LOST", (rnd.shape[1]//4, rnd.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2)
        combo = np.hstack([np.asarray(rgb), np.full((H, 4, 3), 255, np.uint8), rnd]).astype(np.uint8)
        cv2.putText(combo, "REAL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combo, "GAUSSIAN(reloc)", (W+14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        if a.width != combo.shape[1]:
            h2 = round(combo.shape[0]*a.width/combo.shape[1])
            combo = cv2.resize(combo, (a.width, h2))
        frames.append(combo)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps)
    print(f"[replay] {len(frames)}프레임, reloc OK {ok} ({100*ok/max(len(frames),1):.0f}%) → {a.out}")


if __name__ == "__main__":
    main()
