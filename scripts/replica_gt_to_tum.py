"""replica_gt_to_tum.py — Replica traj.txt(c2w 4x4 flatten) → TUM groundtruth.txt.

Replica GT 각 줄 = camera-to-world 4x4 (Twc, SplaTAM reader 규약). timestamp=frame idx.
출력 TUM 포맷: "ts tx ty tz qx qy qz qw" — 07_evaluate.py --gt 로 그대로 먹임.
(ATE는 center=tx ty tz 만 사용하나, quat도 정석대로 c2w rotation에서.)

usage: python scripts/replica_gt_to_tum.py <traj.txt> <out groundtruth.txt>
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(dst, "w") as f:
    for i, line in enumerate(open(src)):
        if not line.strip():
            continue
        m = np.array(list(map(float, line.split()))).reshape(4, 4)  # c2w = Twc
        t = m[:3, 3]
        q = Rot.from_matrix(m[:3, :3]).as_quat()  # xyzw
        f.write(f"{float(i):.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        n += 1
print(f"[gt2tum] {n} poses → {dst}")
