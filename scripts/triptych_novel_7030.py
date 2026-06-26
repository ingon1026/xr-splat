#!/usr/bin/env python3
"""triptych_novel_7030.py — 새 쿼리 2~3장 [real | 거친포즈 렌더 | 정밀화 렌더] 합성.

eval_novel_7030.py의 per-frame 결과(json)에서 PSNR 개선이 뚜렷한 query를 골라,
거친 reloc 포즈와 정밀화 후 포즈로 각각 렌더해 실제 이미지와 나란히 붙인다.

usage: python triptych_novel_7030.py [--n 3] [--tag mcmc2m]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.gsplat_io import load_ply  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402
from scripts.photometric_reloc import se3_exp, render_at  # noqa: E402

DEV = "cuda"


def vm_from_qt(qt):
    qw, qx, qy, qz, tx, ty, tz = qt
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    vm[:3, 3] = [tx, ty, tz]
    return vm


def render_np(g, vm, K, W, H):
    with torch.no_grad():
        im = render_at(g, torch.tensor(vm, device=DEV), K, W, H).clamp(0, 1).cpu().numpy()
    return (im[:, :, ::-1] * 255).astype(np.uint8)


def refine_vm(g, vm_init, gt_t, K, W, H, iters=300):
    base = torch.tensor(vm_init, device=DEV)
    xi = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=3e-3)
    for _ in range(iters):
        vm = se3_exp(xi) @ base
        loss = F.l1_loss(render_at(g, vm, K, W, H).clamp(0, 1), gt_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (se3_exp(xi) @ base).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-full", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--train-scene", default="ros2_bag2_home_rgbd_orbframe_train70")
    ap.add_argument("--tag", default="mcmc2m")
    ap.add_argument("--reloc", default="outputs/ros2_bag2_home_rgbd_orbframe/reloc7030")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", default="docs/assets/report/novel7030_AB.png")
    args = ap.parse_args()

    full = ROOT / "data/processed" / args.scene_full
    reloc = ROOT / args.reloc
    gdir = ROOT / "outputs" / args.train_scene / f"gsplat_{args.tag}"
    sp = full / "colmap/sparse/0"
    W0, H0, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    s = args.scale; W, H = int(W0 * s), int(H0 * s)
    K = torch.tensor([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]], device=DEV)

    imgs_full = read_colmap_images(sp / "images.txt")
    imgs_reg = read_colmap_images(reloc / "registered/images.txt")
    ev = json.loads((gdir / "novel_7030_eval.json").read_text())
    # PSNR 개선 큰 순으로 정렬
    pf = sorted(ev["per_frame"], key=lambda r: r["psnr_refined"] - r["psnr_coarse"], reverse=True)
    pick = [r["name"] for r in pf if r["name"] in imgs_reg][:args.n]

    g = load_ply(gdir / "scene.ply", DEV)
    rows = []
    for n in pick:
        gt = cv2.resize(cv2.imread(str(full / "rgb" / n)), (W, H))
        gt_t = torch.tensor((gt[:, :, ::-1].astype(np.float32) / 255.0).copy(), device=DEV)
        vm_init = vm_from_qt(imgs_reg[n]).astype(np.float32)
        rc = render_np(g, vm_init, K, W, H)
        vm_fin = refine_vm(g, vm_init, gt_t, K, W, H)
        rf = render_np(g, vm_fin, K, W, H)
        rows.append(np.hstack([gt, rc, rf]))
        print(f"  triptych {n}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"[triptych] {len(rows)}행 [real|거친|정밀] → {out}")


if __name__ == "__main__":
    main()
