#!/usr/bin/env python3
"""finalize_orbba_scene.py — Strategy A 마무리: BA정제 모델을 ORB 프레임/스케일로 스냅 + 학습용 씬 구성.

재투영 BA는 절대 스케일이 자유(null-space)라 ~10% 표류. ORB는 depth 기반 metric 진짜값 →
ORB 키프레임에 맞춘 Sim3(near-identity, 잔차~cm)를 BA모델(포즈+점)에 적용해 ORB metric 프레임으로 스냅.
이후 점을 카메라 도달거리로 필터(scene_scale 발산 방지) → data/processed/<scene>_orbba/ 구성.

usage: finalize_orbba_scene.py --scene ros2_bag2_home_rgbd --ba-model <ba_final dir> [--max-reach 6.0]
"""
import argparse, json, sys, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.backproject import read_colmap_images, read_colmap_cameras, read_points3d, colmap_world_RT  # noqa: E402
from pipeline.colmap_convert import write_cameras_txt, write_images_txt  # noqa: E402


def umeyama(src, dst):  # scripts/07_evaluate.py 동일
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Sc.T @ Dc / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / ((Sc ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ba-model", required=True, type=Path, help="ba_final dir (images.txt/points3D.txt)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--max-reach", type=float, default=6.0, help="카메라 중심에서 이 거리[m] 넘는 점 제거")
    ap.add_argument("--out-suffix", default="orbba", help="출력 씬 접미사 <scene>_<suffix>")
    args = ap.parse_args()

    src_proc = args.root / "data" / "processed" / args.scene
    orb_imgs = read_colmap_images(src_proc / "colmap" / "sparse" / "0" / "images.txt")  # 원본 ORB Tcw
    ba_imgs = read_colmap_images(args.ba_model / "images.txt")
    common = sorted(set(orb_imgs) & set(ba_imgs), key=lambda n: float(n.rsplit(".", 1)[0]))

    C_orb = np.array([colmap_world_RT(*orb_imgs[n])[1] for n in common])
    C_ba = np.array([colmap_world_RT(*ba_imgs[n])[1] for n in common])
    s, R, t = umeyama(C_ba, C_orb)                       # BA → ORB
    print(f"[final] Sim3 BA→ORB: scale={s:.5f} rot={np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))):.3f}°")

    # 포즈 스냅: C'=sRC+t, R_wc'=R R_wc → Tcw'
    entries = []
    for i, n in enumerate(common):
        R_wc, C = colmap_world_RT(*ba_imgs[n])
        C2 = s * (R @ C) + t
        R_wc2 = R @ R_wc
        R_cw2 = R_wc2.T
        q = Rot.from_matrix(R_cw2).as_quat()             # (x,y,z,w)
        t_cw2 = -R_cw2 @ C2
        entries.append((i + 1, (q[3], q[0], q[1], q[2], *t_cw2), n))

    # 점 스냅 + 카메라 도달거리 필터
    xyz, rgb = read_points3d(args.ba_model / "points3D.txt")
    xyz2 = (s * (R @ xyz.T).T + t)
    Cs = s * (R @ C_ba.T).T + t                          # ORB 프레임 카메라 중심
    from scipy.spatial import cKDTree
    dmin = cKDTree(Cs).query(xyz2, k=1)[0]
    keep = dmin < args.max_reach
    xyz2, rgb2 = xyz2[keep], rgb[keep]
    scene_scale = float(np.linalg.norm(xyz2 - xyz2.mean(0), axis=1).max())
    print(f"[final] 점 {len(xyz)} → 필터 후 {len(xyz2)} (reach<{args.max_reach}m)  scene_scale={scene_scale:.2f}m")

    # 씬 디렉토리 구성
    dst = args.root / "data" / "processed" / f"{args.scene}_{args.out_suffix}"
    sp = dst / "colmap" / "sparse" / "0"; sp.mkdir(parents=True, exist_ok=True)
    W, H, fx, fy, cx, cy = read_colmap_cameras(src_proc / "colmap" / "sparse" / "0" / "cameras.txt")
    write_cameras_txt(sp / "cameras.txt", W, H, fx, fy, cx, cy)
    write_images_txt(sp / "images.txt", entries)
    with open(sp / "points3D.txt", "w") as f:
        f.write("# 3D point list: POINT3D_ID X Y Z R G B ERROR\n")
        for j, (p, c) in enumerate(zip(xyz2, rgb2)):
            f.write(f"{j+1} {p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} 0\n")

    # rgb/depth 심볼릭, intrinsics/rgb.txt 복사
    for sub in ("rgb", "depth"):
        link = dst / sub
        if not link.exists():
            os.symlink((src_proc / sub).resolve(), link)
    for fn in ("intrinsics.json", "rgb.txt", "depth.txt", "associations.txt"):
        if (src_proc / fn).exists():
            (dst / fn).write_text((src_proc / fn).read_text())
    print(f"[final] → {dst}  (이미지 {len(entries)}, 점 {len(xyz2)})")


if __name__ == "__main__":
    main()
