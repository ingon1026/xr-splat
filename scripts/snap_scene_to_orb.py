#!/usr/bin/env python3
"""snap_scene_to_orb.py — 어긋난 프레임의 씬(예: sfmsnap)을 ORB metric 프레임으로 (재)정렬.

src 씬의 포즈가 ref(ORB) 프레임에서 회전/스케일 어긋났을 때 사용. 두 씬의 공통 프레임(같은 순간)으로
Sim3(src→ref)를 적합한 뒤, **src의 전 프레임 포즈 + 전 포인트**에 적용해 ref 프레임 씬을 새로 만든다.
finalize_orbba_scene과 달리 공통 프레임만이 아니라 src 전체를 보존한다(커버리지 유지).

usage: snap_scene_to_orb.py --src ros2_bag2_home_rgbd_sfmsnap --ref ros2_bag2_home_rgbd --out-suffix orbframe
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.backproject import read_colmap_images, read_colmap_cameras, read_points3d, colmap_world_RT  # noqa: E402
from pipeline.colmap_convert import write_cameras_txt, write_images_txt  # noqa: E402


def umeyama(src, dst):  # finalize_orbba_scene / 07_evaluate 동일
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Sc.T @ Dc / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / ((Sc ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="어긋난 씬 (processed/<src>/colmap/...)")
    ap.add_argument("--ref", required=True, help="ORB 프레임 기준 씬 (processed/<ref>/colmap/...)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-suffix", default="orbframe", help="출력 씬 <ref>_<suffix>")
    args = ap.parse_args()

    src_proc = args.root / "data" / "processed" / args.src
    ref_proc = args.root / "data" / "processed" / args.ref
    src_imgs = read_colmap_images(src_proc / "colmap" / "sparse" / "0" / "images.txt")
    ref_imgs = read_colmap_images(ref_proc / "colmap" / "sparse" / "0" / "images.txt")
    common = sorted(set(src_imgs) & set(ref_imgs), key=lambda n: float(n.rsplit(".", 1)[0]))
    if len(common) < 8:
        sys.exit(f"공통 프레임 {len(common)}개 — Sim3 적합 불가(>=8 필요)")

    C_src = np.array([colmap_world_RT(*src_imgs[n])[1] for n in common])
    C_ref = np.array([colmap_world_RT(*ref_imgs[n])[1] for n in common])
    s, R, t = umeyama(C_src, C_ref)                          # src → ref
    resid = np.linalg.norm((s * (R @ C_src.T).T + t) - C_ref, axis=1) * 100
    print(f"[snap] Sim3 src→ref (공통 {len(common)}): scale={s:.4f} "
          f"rot={np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))):.2f}° "
          f"잔차 median {np.median(resid):.1f}cm p90 {np.percentile(resid,90):.1f}cm")

    # 전 프레임 포즈 스냅: C'=sRC+t, R_wc'=R R_wc → Tcw'
    names = sorted(src_imgs, key=lambda n: float(n.rsplit(".", 1)[0]))
    entries = []
    for i, n in enumerate(names):
        R_wc, C = colmap_world_RT(*src_imgs[n])
        C2 = s * (R @ C) + t
        R_cw2 = (R @ R_wc).T
        q = Rot.from_matrix(R_cw2).as_quat()                 # (x,y,z,w)
        entries.append((i + 1, (q[3], q[0], q[1], q[2], *(-R_cw2 @ C2)), n))

    # 전 포인트 스냅
    xyz, rgb = read_points3d(src_proc / "colmap" / "sparse" / "0" / "points3D.txt")
    xyz2 = s * (R @ xyz.T).T + t

    dst = args.root / "data" / "processed" / f"{args.ref}_{args.out_suffix}"
    sp = dst / "colmap" / "sparse" / "0"; sp.mkdir(parents=True, exist_ok=True)
    W, H, fx, fy, cx, cy = read_colmap_cameras(src_proc / "colmap" / "sparse" / "0" / "cameras.txt")
    write_cameras_txt(sp / "cameras.txt", W, H, fx, fy, cx, cy)
    write_images_txt(sp / "images.txt", entries)
    with open(sp / "points3D.txt", "w") as f:
        f.write("# 3D point list: POINT3D_ID X Y Z R G B ERROR\n")
        for j, (p, c) in enumerate(zip(xyz2, rgb)):
            f.write(f"{j+1} {p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} 0\n")

    for sub in ("rgb", "depth"):
        link = dst / sub
        if not link.exists():
            os.symlink((src_proc / sub).resolve(), link)
    for fn in ("intrinsics.json", "rgb.txt", "depth.txt", "associations.txt"):
        if (src_proc / fn).exists():
            (dst / fn).write_text((src_proc / fn).read_text())
    print(f"[snap] → {dst}  (이미지 {len(entries)}, 점 {len(xyz2)})")


if __name__ == "__main__":
    main()
