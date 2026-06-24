#!/usr/bin/env python3
"""_perturb_images.py — 05 음성대조용: colmap images.txt(Tcw) 포즈를 회전+TZ로 섭동.

05의 측정은 COLMAP Tcw 역투영이므로(마스크만 KeyFrameTrajectory), 게이트가 살아있는지 보려면
**images.txt를 섭동**해야 한다(05 독스트링: TZ/회전 섭동 → FAIL). in-plane translation은 NN거리에 둔감하므로 부적합.
각 포즈를 카메라 프레임에서 rot-deg 회전 + 시선축(+Z)으로 tz-m 이동.

usage: _perturb_images.py --in <images.txt> --out <images.txt> [--rot-deg 3] [--tz-m 0.10] [--seed 0]
"""
import argparse, numpy as np
from scipy.spatial.transform import Rotation as Rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot-deg", type=float, default=3.0)
    ap.add_argument("--tz-m", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out_lines, n = [], 0
    for line in open(args.inp):
        s = line.split()
        if line.startswith("#") or len(s) < 10:
            out_lines.append(line.rstrip("\n")); continue
        iid, (qw, qx, qy, qz), t = s[0], map(float, s[1:5]), np.array(list(map(float, s[5:8])))
        cam, name = s[8], s[9]
        R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        R_wc = R_cw.T
        C = -R_wc @ t                                   # 카메라 중심(월드)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        dR = Rot.from_rotvec(axis * np.radians(args.rot_deg)).as_matrix()
        R_wc2 = R_wc @ dR                               # 카메라 프레임 회전
        C2 = C + R_wc[:, 2] * args.tz_m                 # 시선축(+Z)으로 이동
        R_cw2 = R_wc2.T
        q = Rot.from_matrix(R_cw2).as_quat()            # (x,y,z,w)
        t2 = -R_cw2 @ C2
        out_lines.append(f"{iid} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} "
                         f"{t2[0]:.9f} {t2[1]:.9f} {t2[2]:.9f} {cam} {name}")
        n += 1
    open(args.out, "w").write("\n".join(out_lines) + "\n")
    print(f"[perturb] {n} 포즈 섭동(rot {args.rot_deg}° + TZ {args.tz_m*100:.0f}cm) → {args.out}")


if __name__ == "__main__":
    main()
