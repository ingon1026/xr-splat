#!/usr/bin/env python3
"""export_reloc_poses.py — COLMAP registered 모델(orbframe 좌표)에서 query 프레임 포즈를 TUM Twc로 추출.
출력: outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt (ts tx ty tz qx qy qz qw state)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pipeline.backproject import read_colmap_images

def colmap_image_to_twc_tum(qw, qx, qy, qz, tx, ty, tz):
    R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()      # Tcw 회전
    C = -R_cw.T @ np.array([tx, ty, tz])                    # 카메라 중심(world)
    q = Rot.from_matrix(R_cw.T).as_quat()                   # R_wc → (x,y,z,w)
    return (*C, *q)

def main():
    reg = ROOT/"outputs/ros2_bag2_home_rgbd/reloc_pnp/registered/images.txt"
    qn = ROOT/"outputs/ros2_bag2_home_rgbd/reloc_pnp/query_names.txt"
    out = ROOT/"outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt"; out.parent.mkdir(parents=True, exist_ok=True)
    imgs = read_colmap_images(reg)                          # {name: (qw,qx,qy,qz,tx,ty,tz)}
    queries = [l.strip() for l in open(qn) if l.strip()]
    n = 0
    with open(out, "w") as f:
        for name in sorted(queries, key=lambda s: float(s.rsplit('.',1)[0])):
            ts = name.rsplit('.',1)[0]
            if name in imgs:                                # 등록 성공
                cx,cy,cz,qx,qy,qz,qw = colmap_image_to_twc_tum(*imgs[name])
                f.write(f"{ts} {cx:.7f} {cy:.7f} {cz:.7f} {qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f} 2\n"); n+=1
            else:                                           # 등록 실패 = LOST
                f.write(f"{ts} 0 0 0 0 0 0 1 3\n")
    print(f"[export] query {len(queries)}, 등록 {n} ({100*n/max(len(queries),1):.0f}%) → {out}")

if __name__ == "__main__":
    main()
