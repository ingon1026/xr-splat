#!/usr/bin/env python3
"""extract_ros2_rgbd.py — ROS2 .db3(RealSense)에서 RGB + depth를 color에 정렬해 TUM RGB-D로 추출.

ORB-SLAM3 RGB-D 입력용. rs.align(depth→color)을 extrinsics로 직접 재구현.
ROS2 환경에서: source /opt/ros/jazzy/setup.bash 후 system python3.

출력: <out>/{rgb,depth}/<ts>.png, rgb.txt, depth.txt, associations.txt, intrinsics.json
usage: extract_ros2_rgbd.py --db3 x.db3 --out data/processed/<scene> [--stride 2]
"""
import argparse, sqlite3, json, bisect
from pathlib import Path
import numpy as np
import cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import String

COLOR = "/device_0/sensor_1/Color_0/image/data"
COLOR_CI = "/device_0/sensor_1/Color_0/camera_info"
DEPTH = "/device_0/sensor_0/Depth_0/image/data"
DEPTH_CI = "/device_0/sensor_0/Depth_0/camera_info"
COLOR_TF = "/device_0/sensor_1/Color_0/tf/ref_0"   # color pose in depth(ref) frame
DEPTH_UNITS = "/device_0/sensor_0/option/Depth_Units/value"


def caminfo(s):
    d = dict(kv.split("=", 1) for kv in s.split(";") if "=" in kv)
    return float(d["fx"]), float(d["fy"]), float(d["ppx"]), float(d["ppy"]), int(d["width"]), int(d["height"]), \
        [float(x) for x in d.get("coeffs", "0,0,0,0,0").split(",")]


def parse_tf(s):
    d = dict(kv.split("=", 1) for kv in s.split(";") if "=" in kv)
    R = np.array([float(x) for x in d["rotation"].split(",")]).reshape(3, 3)
    t = np.array([float(x) for x in d["translation"].split(",")])
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db3", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    (args.out / "rgb").mkdir(parents=True, exist_ok=True)
    (args.out / "depth").mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.db3}?mode=ro", uri=True)
    cur = con.cursor()
    g1 = lambda n, T: deserialize_message(
        cur.execute("SELECT data FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name=? LIMIT 1", (n,)).fetchone()[0], T)
    fxc, fyc, cxc, cyc, Wc, Hc, _ = caminfo(g1(COLOR_CI, String).data)
    fxd, fyd, cxd, cyd, Wd, Hd, _ = caminfo(g1(DEPTH_CI, String).data)
    Rc, tc = parse_tf(g1(COLOR_TF, String).data)            # color in depth frame: p_d = Rc@p_c + tc
    du = float(g1(DEPTH_UNITS, String).data)                # m per raw unit (0.001)
    tid = lambda n: cur.execute("SELECT id FROM topics WHERE name=?", (n,)).fetchone()[0]

    # depth 인덱스 (timestamp, rowid)
    dlist = cur.execute("SELECT timestamp, id FROM messages WHERE topic_id=? ORDER BY timestamp", (tid(DEPTH),)).fetchall()
    dts = [r[0] for r in dlist]
    # depth deproject 그리드 (depth 픽셀 → 카메라 광선)
    uu, vv = np.meshgrid(np.arange(Wd), np.arange(Hd))
    xn = (uu - cxd) / fxd; yn = (vv - cyd) / fyd            # (Hd,Wd)

    pairs = []
    crows = cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid(COLOR),)).fetchall()
    for i, (cts, cblob) in enumerate(crows):
        if i % args.stride:
            continue
        cimg = deserialize_message(cblob, Image)
        ts = cimg.header.stamp.sec + cimg.header.stamp.nanosec * 1e-9
        color = np.frombuffer(cimg.data, np.uint8).reshape(Hc, Wc, 3)[:, :, ::-1]  # rgb→bgr
        # 가장 가까운 depth
        j = bisect.bisect_left(dts, cts); j = min(range(max(0, j - 1), min(len(dts), j + 1)), key=lambda k: abs(dts[k] - cts))
        dblob = cur.execute("SELECT data FROM messages WHERE id=?", (dlist[j][1],)).fetchone()[0]
        Z = np.frombuffer(deserialize_message(dblob, Image).data, np.uint16).reshape(Hd, Wd).astype(np.float32) * du  # meters
        # deproject(depth) → depth frame 3D → color frame → project(color)
        pd = np.stack([xn * Z, yn * Z, Z], -1).reshape(-1, 3)          # (N,3) depth frame
        pc = (pd - tc) @ Rc                                            # p_c = Rc^T@(p_d - tc)  (행벡터)
        m = pc[:, 2] > 0
        uc = (fxc * pc[:, 0] / pc[:, 2] + cxc).round().astype(int)
        vc = (fyc * pc[:, 1] / pc[:, 2] + cyc).round().astype(int)
        ok = m & (uc >= 0) & (uc < Wc) & (vc >= 0) & (vc < Hc)
        flat = np.full(Wc * Hc, np.inf, np.float32)                  # z-buffer(최근접) 벡터화
        np.minimum.at(flat, vc[ok] * Wc + uc[ok], pc[:, 2][ok])
        flat[np.isinf(flat)] = 0
        depth_mm = np.clip(flat.reshape(Hc, Wc) * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(args.out / f"rgb/{ts:.6f}.png"), color)
        cv2.imwrite(str(args.out / f"depth/{ts:.6f}.png"), depth_mm)
        pairs.append((ts, f"rgb/{ts:.6f}.png", f"depth/{ts:.6f}.png"))
        if len(pairs) % 100 == 0:
            print(f"  {len(pairs)} frames...")
    print(f"[rgbd] {len(pairs)} frames  color {Wc}x{Hc}  depth aligned→color")

    intr = dict(fx=fxc, fy=fyc, cx=cxc, cy=cyc, width=Wc, height=Hc, depth_scale=1000.0,
                distortion=dict(model="inverse_brown_conrady", k1=0, k2=0, p1=0, p2=0, k3=0),
                source=f"ros2:{Path(args.db3).name}", model="PINHOLE")
    (args.out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
    with open(args.out / "rgb.txt", "w") as fr, open(args.out / "depth.txt", "w") as fd, \
         open(args.out / "associations.txt", "w") as fa:
        fr.write("# timestamp filename\n"); fd.write("# timestamp filename\n")
        for ts, r, d in pairs:
            fr.write(f"{ts:.6f} {r}\n"); fd.write(f"{ts:.6f} {d}\n"); fa.write(f"{ts:.6f} {r} {ts:.6f} {d}\n")
    print(f"[rgbd] → {args.out}")


if __name__ == "__main__":
    main()
