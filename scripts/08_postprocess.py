#!/usr/bin/env python3
"""08_postprocess.py — 후처리: opacity pruning → SH degree 3→1 축소 → 경량 export.

각 단계 전후의 (가우시안 수, 파일 크기, PSNR) 표를 자동 출력. XR 배포용 경량화.
PSNR은 고정 키프레임 뷰 렌더 vs GT (품질 저하 추적).

usage: 08_postprocess.py --scene <scene> [--tag ""] [--opacity 0.05] [--sh-out 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, save_ply, render  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402

DEV = "cuda"


def view_psnr(g, proc, name, K, W, H):
    qw, qx, qy, qz, tx, ty, tz = read_colmap_images(proc / "colmap" / "sparse" / "0" / "images.txt")[name]
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    vm[:3, 3] = [tx, ty, tz]
    pred = render(g, torch.tensor(vm, device=DEV), K, W, H).detach().cpu().numpy()
    gt = cv2.imread(str(proc / "rgb" / name))[:, :, ::-1].astype(np.float32) / 255.0
    return float(-10 * np.log10(max(((pred - gt) ** 2).mean(), 1e-10)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--tag", default="")
    ap.add_argument("--opacity", type=float, default=0.05, help="opacity prune 임계 (sigmoid)")
    ap.add_argument("--sh-out", type=int, default=1, help="축소 후 SH degree")
    args = ap.parse_args()
    proc = args.root / "data" / "processed" / args.scene
    gdir = args.root / "outputs" / args.scene / ("gsplat" + (f"_{args.tag}" if args.tag else ""))
    src = gdir / "scene.ply"
    W, H, fx, fy, cx, cy = read_colmap_cameras(proc / "colmap" / "sparse" / "0" / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    ref = sorted(read_colmap_images(proc / "colmap" / "sparse" / "0" / "images.txt").keys())[0]

    rows = []

    def record(stage, g, path):
        size_mb = Path(path).stat().st_size / 1e6
        rows.append(dict(stage=stage, n=int(g["means"].shape[0]),
                         size_mb=round(size_mb, 1), psnr=round(view_psnr(g, proc, ref, K, W, H), 2)))

    g = load_ply(src, DEV)
    record("0_original", g, src)

    # 1) opacity pruning
    keep = torch.sigmoid(g["opacities"]) >= args.opacity
    g1 = {k: (v[keep] if k != "sh_degree" else v) for k, v in g.items()}
    p1 = gdir / "scene_pruned.ply"; save_ply(p1, g1)
    record("1_opacity_prune", g1, p1)

    # 2) SH degree 3→sh_out 축소
    nkeep = (args.sh_out + 1) ** 2
    g2 = dict(g1); g2["sh"] = g1["sh"][:, :nkeep, :]; g2["sh_degree"] = args.sh_out
    p2 = gdir / "scene_shreduced.ply"; save_ply(p2, g2)
    record(f"2_sh_3to{args.sh_out}", g2, p2)

    # 3) 경량 export (prune+SH축소 결과 = 최종)
    final = gdir / "scene_lite.ply"; save_ply(final, g2)
    record("3_export_lite", g2, final)

    out = dict(ref_view=ref, opacity_thresh=args.opacity, sh_out=args.sh_out, stages=rows)
    (gdir / "postprocess.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n===== 후처리 표 (scene={args.scene}, ref={ref}) =====")
    print(f"{'stage':<18}{'gaussians':>12}{'size(MB)':>12}{'PSNR':>9}")
    for r in rows:
        print(f"{r['stage']:<18}{r['n']:>12,}{r['size_mb']:>12}{r['psnr']:>9}")
    print(f"\n최종 경량 모델 → {final}")


if __name__ == "__main__":
    main()
