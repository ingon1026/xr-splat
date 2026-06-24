#!/usr/bin/env python3
"""eval_sfmsnap_gap.py — sfmsnap이 base(ORB) 대비 렌더 품질 격차를 메꾸는지 holdout PSNR로 확인.

sfmsnap 씬은 28뷰 holdout으로 학습됨(엄밀). base/sfm은 full train(holdout 없음)이라
같은 28 프레임에 렌더하면 'train-view 낙관치'다 — 그래도 base가 거기서조차 낮으면 격차는 진짜.
각 씬은 자기 colmap 포즈(images.txt)로 렌더, GT는 동일 rgb.

usage: eval_sfmsnap_gap.py --capture room2   # room2|home
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, render            # noqa: E402
from pipeline.backproject import read_colmap_cameras, read_colmap_images  # noqa: E402

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[1]


def eval_scene(scene, ply, holdout, label):
    """scene의 colmap 포즈로 holdout 프레임 렌더 → PSNR 리스트. (없는 프레임은 skip)"""
    proc = ROOT / "data" / "processed" / scene
    sp = proc / "colmap" / "sparse" / "0"
    if not ply.exists():
        print(f"  [{label}] ply 없음: {ply}"); return None
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEV)
    imgs = read_colmap_images(sp / "images.txt")
    g = load_ply(ply, DEV)
    ps, miss = [], 0
    for name in holdout:
        if name not in imgs:
            miss += 1; continue
        qw, qx, qy, qz, tx, ty, tz = imgs[name]
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix(); vm[:3, 3] = [tx, ty, tz]
        pred = render(g, torch.tensor(vm, device=DEV), K, W, H).detach().cpu().numpy()
        gt = cv2.imread(str(proc / "rgb" / name))[:, :, ::-1].astype(np.float32) / 255.0
        ps.append(float(-10 * np.log10(max(((pred - gt) ** 2).mean(), 1e-10))))
    if not ps:
        print(f"  [{label}] 공통 프레임 0 (miss {miss}) — 비교 불가"); return None
    arr = np.array(ps)
    print(f"  [{label:22s}] n={len(ps):2d}(miss {miss:2d})  PSNR median {np.median(arr):.2f}  mean {arr.mean():.2f}  min {arr.min():.2f}")
    return dict(label=label, scene=scene, n=len(ps), psnr_median=round(float(np.median(arr)), 2),
                psnr_mean=round(float(arr.mean()), 2), psnr_min=round(float(arr.min()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, choices=["room2", "home"])
    args = ap.parse_args()
    if args.capture == "room2":
        snap, base, sfm = "d455_room2_sfmsnap", "d455_room2", "d455_room2_sfm"
        plys = {"sfmsnap (held-out, 엄밀)": (snap, ROOT/f"outputs/{snap}/gsplat_sfmsnap/scene.ply"),
                "base ORB (train-view, 낙관)": (base, ROOT/f"outputs/{base}/gsplat/scene.ply"),
                "sfm (train-view, 낙관)": (sfm, ROOT/f"outputs/{sfm}/gsplat_v2/scene.ply")}
    else:
        snap, base, sfm = "ros2_bag2_home_rgbd_sfmsnap", "ros2_bag2_home_rgbd", "ros2_bag2_home_sfm"
        plys = {"sfmsnap (held-out, 엄밀)": (snap, ROOT/f"outputs/{snap}/gsplat_sfmsnap/scene.ply"),
                "base ORB (train-view, 낙관)": (base, ROOT/f"outputs/{base}/gsplat/scene.ply"),
                "sfm (train-view, 낙관)": (sfm, ROOT/f"outputs/{sfm}/gsplat_v1/scene.ply")}

    holdout = [l.strip() for l in open(ROOT/f"outputs/{snap}/gsplat_sfmsnap/holdout.txt") if l.strip()]
    print(f"=== {args.capture}: holdout {len(holdout)}뷰 (sfmsnap 학습 제외분) ===")
    out = []
    for label, (scene, ply) in plys.items():
        r = eval_scene(scene, ply, holdout, label)
        if r:
            out.append(r)
    (ROOT/f"outputs/{snap}/gap_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"=== 저장: outputs/{snap}/gap_eval.json ===")


if __name__ == "__main__":
    main()
