#!/usr/bin/env python3
"""eval_novel_7030.py — 70/30 진짜 새 시점 end-to-end 평가.

뒤 30% KF(가우시안 학습·ref 모델 둘 다에서 빠진 새 시점)에 대해:
  init = image_registrator가 준 **실제 거친 reloc 포즈**(reloc7030/registered)
  → 70% 가우시안(train70 자산) 대상 photometric 정밀화(se3 delta, Adam, L1)
  측정:
    - 거친 포즈오차 / 정밀화 후 포즈오차 (기준=전체 모델의 그 KF 포즈, true GT 아님·최선 reference)
    - render-vs-real PSNR: 거친 포즈 vs 정밀화 후 (1차 지표=이미지가 정답)
    - 새로움 정량화: 각 query 카메라중심 → 최근접 ref 카메라중심 거리(cm)

usage: python eval_novel_7030.py [--tag mcmc2m] [--iters 300] [--scale 0.5] [--frames 0(=all)]
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
from scripts.photometric_reloc import se3_exp, render_at, pose_err  # noqa: E402

DEV = "cuda"


def vm_from_qt(qt):
    """(qw,qx,qy,qz,tx,ty,tz) → Tcw viewmat 4x4."""
    qw, qx, qy, qz, tx, ty, tz = qt
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    vm[:3, 3] = [tx, ty, tz]
    return vm


def cam_center(vm):
    return np.linalg.inv(vm)[:3, 3]


def relocalize(g, vm_init, gt_t, K, W, H, iters):
    """vm_init에서 시작, render가 gt와 맞도록 se3 delta 최적화. 최종 viewmat + PSNR."""
    base = torch.tensor(vm_init, device=DEV)
    xi = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=3e-3)
    for _ in range(iters):
        vm = se3_exp(xi) @ base
        pred = render_at(g, vm, K, W, H).clamp(0, 1)
        loss = F.l1_loss(pred, gt_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        vm_fin = (se3_exp(xi) @ base).detach().cpu().numpy()
        psnr = float(-10 * np.log10(max(F.mse_loss(
            render_at(g, torch.tensor(vm_fin, device=DEV), K, W, H).clamp(0, 1), gt_t).item(), 1e-10)))
    return vm_fin, psnr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-full", default="ros2_bag2_home_rgbd_orbframe",
                    help="기준 KF 포즈·rgb 원본 scene")
    ap.add_argument("--train-scene", default="ros2_bag2_home_rgbd_orbframe_train70",
                    help="70% 가우시안 학습 scene")
    ap.add_argument("--tag", default="mcmc2m")
    ap.add_argument("--reloc", default="outputs/ros2_bag2_home_rgbd_orbframe/reloc7030")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=0, help="0=등록된 query 전부")
    args = ap.parse_args()

    full = ROOT / "data/processed" / args.scene_full
    reloc = ROOT / args.reloc
    gdir = ROOT / "outputs" / args.train_scene / f"gsplat_{args.tag}"

    sp_full = full / "colmap/sparse/0"
    W0, H0, fx, fy, cx, cy = read_colmap_cameras(sp_full / "cameras.txt")
    s = args.scale
    W, H = int(W0 * s), int(H0 * s)
    K = torch.tensor([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]], device=DEV)

    imgs_full = read_colmap_images(sp_full / "images.txt")        # 기준 포즈(전체 모델)
    imgs_reg = read_colmap_images(reloc / "registered/images.txt")  # 거친 reloc 포즈
    query_names = [l.strip() for l in open(reloc / "query_names.txt") if l.strip()]
    ref_names = [l.strip() for l in open(reloc / "ref_names.txt") if l.strip()]

    # 등록 성공 = registered 모델에 있는 query
    reg_query = [n for n in query_names if n in imgs_reg]
    print(f"[novel] query {len(query_names)}, 등록 {len(reg_query)}", flush=True)
    if args.frames > 0:
        reg_query = reg_query[:: max(1, len(reg_query) // args.frames)][:args.frames]

    # ref 카메라중심(새로움 거리용)
    ref_centers = np.array([cam_center(vm_from_qt(imgs_full[n])) for n in ref_names if n in imgs_full])

    g = load_ply(gdir / "scene.ply", DEV)
    print(f"[novel] gaussians {g['means'].shape[0]}, {W}x{H}, asset={args.train_scene}/{args.tag}, iters={args.iters}", flush=True)

    res = []
    for n in reg_query:
        vm_ref = vm_from_qt(imgs_full[n])        # 기준(전체 모델 KF 포즈)
        vm_init = vm_from_qt(imgs_reg[n]).astype(np.float32)  # 거친 reloc 포즈

        gt = cv2.resize(cv2.imread(str(full / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0, (W, H))
        gt_t = torch.tensor(gt.copy(), device=DEV)
        psnr0 = float(-10 * np.log10(max(F.mse_loss(
            render_at(g, torch.tensor(vm_init, device=DEV), K, W, H).clamp(0, 1), gt_t).item(), 1e-10)))
        dt0, dr0 = pose_err(vm_init, vm_ref)

        vm_fin, psnr1 = relocalize(g, vm_init, gt_t, K, W, H, args.iters)
        dt1, dr1 = pose_err(vm_fin, vm_ref)

        novelty = float(np.linalg.norm(ref_centers - cam_center(vm_ref), axis=1).min() * 100)
        res.append(dict(name=n, novelty_cm=round(novelty, 1),
                        coarse_trans_cm=round(dt0, 2), coarse_rot_deg=round(dr0, 2),
                        refined_trans_cm=round(dt1, 2), refined_rot_deg=round(dr1, 2),
                        psnr_coarse=round(psnr0, 2), psnr_refined=round(psnr1, 2)))
        print(f"  {n}: novelty {novelty:.1f}cm | pose {dt0:.1f}cm/{dr0:.1f}° → {dt1:.1f}cm/{dr1:.1f}° | "
              f"PSNR {psnr0:.1f} → {psnr1:.1f}", flush=True)

    if not res:
        print("[novel] 등록된 query 0개 — eval 불가", flush=True); return
    a = lambda k: np.array([r[k] for r in res])
    summ = dict(
        train_scene=args.train_scene, tag=args.tag, iters=args.iters,
        n_query=len(query_names), n_registered=len([n for n in query_names if n in imgs_reg]),
        n_evaluated=len(res),
        novelty_cm=dict(median=round(float(np.median(a("novelty_cm"))), 1),
                        min=round(float(a("novelty_cm").min()), 1),
                        max=round(float(a("novelty_cm").max()), 1)),
        pose_trans_cm=dict(coarse=round(float(np.median(a("coarse_trans_cm"))), 2),
                           refined=round(float(np.median(a("refined_trans_cm"))), 2)),
        pose_rot_deg=dict(coarse=round(float(np.median(a("coarse_rot_deg"))), 2),
                          refined=round(float(np.median(a("refined_rot_deg"))), 2)),
        psnr=dict(coarse=round(float(np.median(a("psnr_coarse"))), 2),
                  refined=round(float(np.median(a("psnr_refined"))), 2)),
        per_frame=res)
    outp = gdir / "novel_7030_eval.json"
    outp.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(f"\n[novel] ===== 요약(median, n={len(res)}) =====", flush=True)
    print(f"  새로움(최근접 ref): {summ['novelty_cm']['median']}cm (min {summ['novelty_cm']['min']}, max {summ['novelty_cm']['max']})", flush=True)
    print(f"  포즈오차: 거친 {summ['pose_trans_cm']['coarse']}cm/{summ['pose_rot_deg']['coarse']}° "
          f"→ 정밀 {summ['pose_trans_cm']['refined']}cm/{summ['pose_rot_deg']['refined']}°", flush=True)
    print(f"  render-vs-real PSNR: 거친 {summ['psnr']['coarse']} → 정밀 {summ['psnr']['refined']}", flush=True)
    print(f"[novel] json → {outp}", flush=True)
    print("[novel] DONE", flush=True)


if __name__ == "__main__":
    main()
