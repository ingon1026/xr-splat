#!/usr/bin/env python3
"""render_teaser.py — 학습된 3DGS scene.ply로 fly-through 티저 GIF 렌더 (M3).

키프레임 포즈(COLMAP Tcw)를 시간순으로 정렬 → 일부를 anchor로 균등 샘플 →
회전 Slerp + 위치 Lerp로 부드러운 경로를 만들어 프레임 렌더 → GIF로 인코딩.
대표 프레임 PNG도 함께 저장(육안 검토용).

usage: render_teaser.py --scene tum_fr1_desk [--tag ""] [--n-anchors 12] [--steps 4]
                        [--width 512] [--fps 18]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, render  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402

DEV = "cuda"


def tcw_to_twc(qw, qx, qy, qz, tx, ty, tz):
    """COLMAP Tcw(world->cam) → camera center C, R_wc(cam->world)."""
    R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    C = -R_cw.T @ np.array([tx, ty, tz])
    return C, R_cw.T


def twc_to_viewmat(C, R_wc):
    """camera center C + R_wc → Tcw 4x4 viewmat(world->cam)."""
    R_cw = R_wc.T
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R_cw
    vm[:3, 3] = -R_cw @ C
    return vm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="tum_fr1_desk")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--tag", default="", help="gsplat 출력 서브디렉토리 접미사 ('' = Phase5 30k 모델)")
    ap.add_argument("--n-anchors", type=int, default=12, help="궤적에서 균등 샘플할 anchor 키프레임 수")
    ap.add_argument("--start-frac", type=float, default=0.0, help="궤적 사용 시작 비율(0~1)")
    ap.add_argument("--end-frac", type=float, default=1.0, help="궤적 사용 끝 비율(0~1) — floater 구간 트리밍용")
    ap.add_argument("--steps", type=int, default=4, help="anchor 사이 보간 프레임 수")
    ap.add_argument("--width", type=int, default=512, help="GIF 가로 픽셀(다운스케일)")
    ap.add_argument("--colors", type=int, default=256, help="GIF 팔레트 색 수(용량 조절)")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    proc = args.root / "data" / "processed" / args.scene
    gdir = args.root / "outputs" / args.scene / ("gsplat" + (f"_{args.tag}" if args.tag else ""))
    sp = proc / "colmap" / "sparse" / "0"
    out_gif = args.out or (args.root / "docs" / "assets" / "teaser.gif")
    frames_dir = args.root / "outputs" / args.scene / "teaser_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    g = load_ply(gdir / "scene.ply", DEV)
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    print(f"[teaser] model={gdir/'scene.ply'} N={g['means'].shape[0]} {W}x{H} sh_deg={g['sh_degree']}")

    imgs = read_colmap_images(sp / "images.txt")
    names = sorted(imgs.keys(), key=lambda n: float(Path(n).stem))
    i0 = int(round(args.start_frac * (len(names) - 1)))
    i1 = int(round(args.end_frac * (len(names) - 1)))
    idx = np.linspace(i0, i1, args.n_anchors).round().astype(int)
    anchors = [names[i] for i in idx]
    Cs = np.array([tcw_to_twc(*imgs[n])[0] for n in anchors])
    Rs = Rot.from_matrix([tcw_to_twc(*imgs[n])[1] for n in anchors])
    key_t = np.arange(len(anchors), dtype=float)
    slerp = Slerp(key_t, Rs)
    print(f"[teaser] anchors={len(anchors)} (of {len(names)} kf), frames={(len(anchors)-1)*args.steps}")

    sample_t = np.linspace(0, len(anchors) - 1, (len(anchors) - 1) * args.steps + 1)
    frames = []
    for k, t in enumerate(sample_t):
        i = min(int(np.floor(t)), len(anchors) - 2)
        a = t - i
        C = (1 - a) * Cs[i] + a * Cs[i + 1]
        R_wc = slerp([t])[0].as_matrix()
        vm = torch.tensor(twc_to_viewmat(C, R_wc), device=DEV)
        with torch.no_grad():
            rgb = render(g, vm, K, W, H).detach().cpu().numpy()
        img = (rgb * 255).astype(np.uint8)  # RGB
        if args.width != W:
            h2 = round(H * args.width / W)
            img = np.array(Image.fromarray(img).resize((args.width, h2), Image.LANCZOS))
        frames.append(img)
        if k % 8 == 0:
            print(f"  frame {k}/{len(sample_t)}")

    # 대표 프레임 PNG (육안 검토): 처음/중간/끝
    reps = [0, len(frames) // 2, len(frames) - 1]
    rep_paths = []
    for j, fi in enumerate(reps):
        p = frames_dir / f"rep_{j}_frame{fi:03d}.png"
        imageio.imwrite(p, frames[fi])
        rep_paths.append(p)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=args.fps, loop=0, palettesize=args.colors, subrectangles=True)
    size_mb = out_gif.stat().st_size / 1e6
    print(f"\n[teaser] GIF → {out_gif} ({len(frames)} frames, {args.width}px, {size_mb:.2f} MB)")
    print(f"[teaser] 대표 PNG: " + ", ".join(str(p) for p in rep_paths))
    if size_mb > 2.0:
        print(f"[teaser] ⚠ {size_mb:.2f}MB > 2MB 목표 초과 — width/steps/n-anchors 낮춰 재실행 필요")


if __name__ == "__main__":
    main()
