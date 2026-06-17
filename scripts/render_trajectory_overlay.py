#!/usr/bin/env python3
"""render_trajectory_overlay.py — 가우시안 씬을 외부 궤도 카메라로 돌며, SLAM 키프레임 궤적+프러스텀을
2D 투영해 렌더 프레임 위에 그려 fly-through GIF 생성 (Part 2, 출력 A — 논문 figure용).

render_teaser.py의 모델/intrinsics 로드·render()·GIF 인코딩을 재사용. 차이점:
  - on-path 카메라(궤적이 등 뒤) 대신 **외부 orbit 카메라**(궤적을 바라봄).
  - 매 프레임 선분 끝점을 풀해상도에서 투영→PIL line→resize.
  - occlusion 없음(선이 splat 위에 항상): teaser 시각화 허용.

usage: render_trajectory_overlay.py --scene <s> --model <scene.ply 경로>
       [--frustum-stride 5] [--frustum-size 1.0] [--width 720] [--frames 120] [--orbit-radius 2.0]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, render  # noqa: E402
from pipeline.backproject import read_colmap_cameras  # noqa: E402
from pipeline.overlay_geometry import build_segments  # noqa: E402

DEV = "cuda"


def look_at(pos, target, world_up):
    z = target - pos; z /= np.linalg.norm(z)               # forward +Z
    x = np.cross(world_up, z); x /= np.linalg.norm(x)      # right +X
    y = np.cross(z, x)                                      # down +Y
    R_wc = np.stack([x, y, z], 1)                           # cam→world (열=카메라축)
    R_cw = R_wc.T
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R_cw; vm[:3, 3] = -R_cw @ pos
    return vm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True, type=Path, help="scene.ply 경로")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--images", type=Path, help="기본: data/processed/<scene>/colmap/sparse/0/images.txt")
    ap.add_argument("--frustum-stride", type=int, default=5)
    ap.add_argument("--frustum-size", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--orbit-radius", type=float, default=2.0, help="궤적 bbox 대각 대비 궤도 반경 배율")
    ap.add_argument("--orbit-deg", type=float, default=360.0)
    ap.add_argument("--elev", type=float, default=0.5, help="반경 대비 상승 높이 배율")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sp = args.root / "data" / "processed" / args.scene / "colmap" / "sparse" / "0"
    images = args.images or (sp / "images.txt")
    g = load_ply(args.model, DEV)
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    segs, d, traj_diag = build_segments(images, sp / "cameras.txt", args.frustum_stride, args.frustum_size)
    print(f"[overlay] N={g['means'].shape[0]} {W}x{H} segs={len(segs)} near_d={d:.3f}")

    centers = np.array([s[0] for s in segs] + [s[1] for s in segs])
    c0 = centers.mean(0)
    diag = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    r = max(args.orbit_radius * diag, 0.5)
    # up = 키프레임 중심이 가장 적게 움직인 월드 축(보통 수직)
    kf_c = np.array([s[0] for s in segs])
    up_axis = int(np.argmin(kf_c.std(0))); world_up = np.zeros(3); world_up[up_axis] = 1.0
    e1 = np.zeros(3); e2 = np.zeros(3)
    others = [a for a in range(3) if a != up_axis]
    e1[others[0]] = 1.0; e2[others[1]] = 1.0

    thetas = np.linspace(0, np.radians(args.orbit_deg), args.frames, endpoint=False)
    frames = []
    for k, th in enumerate(thetas):
        pos = c0 + r * (np.cos(th) * e1 + np.sin(th) * e2) + args.elev * r * world_up
        vm_np = look_at(pos, c0, world_up)
        vm = torch.tensor(vm_np, device=DEV)
        with torch.no_grad():
            rgb = render(g, vm, K, W, H).detach().cpu().numpy()
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        R_cw, t = vm_np[:3, :3], vm_np[:3, 3]
        for P0, P1, col in segs:
            a = R_cw @ P0 + t; b = R_cw @ P1 + t
            if a[2] <= 1e-3 or b[2] <= 1e-3:
                continue
            ua, va = fx * a[0] / a[2] + cx, fy * a[1] / a[2] + cy
            ub, vb = fx * b[0] / b[2] + cx, fy * b[1] / b[2] + cy
            draw.line([(ua, va), (ub, vb)], fill=tuple(int(c * 255) for c in col), width=2)
        if args.width != W:
            h2 = round(H * args.width / W)
            img = img.resize((args.width, h2), Image.LANCZOS)
        frames.append(np.array(img))
        if k % 20 == 0:
            print(f"  frame {k}/{len(thetas)}")

    out_gif = args.out or (args.root / "docs" / "assets" / f"trajectory_overlay_{args.scene}.gif")
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    fdir = args.root / "outputs" / args.scene / "trajectory_frames"; fdir.mkdir(parents=True, exist_ok=True)
    for j, fi in enumerate([0, len(frames) // 3, 2 * len(frames) // 3]):
        imageio.imwrite(fdir / f"rep_{j}.png", frames[fi])
    imageio.mimsave(out_gif, frames, fps=args.fps, loop=0, subrectangles=True)
    print(f"[overlay] GIF → {out_gif} ({len(frames)}f, {out_gif.stat().st_size/1e6:.2f}MB)  rep → {fdir}")


if __name__ == "__main__":
    main()
