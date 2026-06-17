#!/usr/bin/env python3
"""eval_holdout_psnr.py — 학습된 scene.ply를 hold-out 키프레임에서 렌더해 PSNR/SSIM 측정 (G3 게이트).

train_gsplat.py가 남긴 holdout.txt(학습 제외 뷰)로 화질을 정량 평가. "화질 손실 없음" 증명용.
usage: eval_holdout_psnr.py --scene <s> --model <scene.ply> [--holdout <holdout.txt>]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation as Rot
from pytorch_msssim import ssim as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, render  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras  # noqa: E402

DEV = "cuda"


def viewmat(qw, qx, qy, qz, tx, ty, tz):
    R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R_cw; vm[:3, 3] = [tx, ty, tz]
    return torch.tensor(vm, device=DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--holdout", type=Path)
    args = ap.parse_args()
    proc = args.root / "data" / "processed" / args.scene
    sp = proc / "colmap" / "sparse" / "0"
    holdout = args.holdout or (args.model.parent / "holdout.txt")

    g = load_ply(args.model, DEV)
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    imgs = read_colmap_images(sp / "images.txt")
    names = [l.strip() for l in open(holdout) if l.strip()]

    psnrs, ssims = [], []
    for n in names:
        if n not in imgs:
            continue
        with torch.no_grad():
            pred = render(g, viewmat(*imgs[n]), K, W, H)
        gt = cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0
        psnr = float(-10 * np.log10(max(((pred.cpu().numpy() - gt) ** 2).mean(), 1e-10)))
        ss = float(ssim_fn(pred.permute(2, 0, 1)[None],
                           torch.tensor(gt.copy(), device=DEV).permute(2, 0, 1)[None], data_range=1.0))
        psnrs.append(psnr); ssims.append(ss)
    psnrs, ssims = np.array(psnrs), np.array(ssims)
    print(f"[g3] {args.scene}  hold-out {len(psnrs)}뷰  PSNR median={np.median(psnrs):.2f} mean={psnrs.mean():.2f}  "
          f"SSIM median={np.median(ssims):.4f}")
    print(f"[g3] PSNR min={psnrs.min():.2f} max={psnrs.max():.2f}")


if __name__ == "__main__":
    main()
