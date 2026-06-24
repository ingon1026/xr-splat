#!/usr/bin/env python3
"""extract_ros2_color.py — ROS2 rosbag2(.db3, RealSense 스키마)에서 COLOR 프레임만 추출.

COLMAP RGB-only 경로용(experiment/colmap-poses). depth/정렬 불필요 → 컬러 Image만 뽑는다.
ROS2 환경에서 실행: source /opt/ros/jazzy/setup.bash 후 system python3.

출력: <out>/rgb/<ts>.png + intrinsics.json + rgb.txt
usage: extract_ros2_color.py --db3 x.db3 --out data/processed/<scene> [--target-frames 300]
"""
import argparse, sqlite3, json
from pathlib import Path
import numpy as np
import cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import String

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
COLOR_CAMINFO = "/device_0/sensor_1/Color_0/camera_info"


def parse_caminfo(s):
    d = dict(kv.split("=", 1) for kv in s.split(";") if "=" in kv)
    coeffs = [float(x) for x in d.get("coeffs", "0,0,0,0,0").split(",")]
    return dict(fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["ppx"]), cy=float(d["ppy"]),
                width=int(d["width"]), height=int(d["height"]),
                model=d.get("model", ""), coeffs=coeffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db3", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-frames", type=int, default=300)
    args = ap.parse_args()
    (args.out / "rgb").mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.db3}?mode=ro", uri=True)
    cur = con.cursor()
    tid = lambda n: cur.execute("SELECT id FROM topics WHERE name=?", (n,)).fetchone()[0]

    ci = deserialize_message(cur.execute(
        "SELECT data FROM messages WHERE topic_id=? LIMIT 1", (tid(COLOR_CAMINFO),)).fetchone()[0], String)
    K = parse_caminfo(ci.data)

    ct = tid(COLOR_TOPIC)
    total = cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (ct,)).fetchone()[0]
    stride = max(1, total // args.target_frames)
    print(f"[ros2] color {total}장, stride={stride} → ~{total // stride}장 추출")

    pairs = []
    for i, (blob,) in enumerate(cur.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (ct,))):
        if i % stride:
            continue
        img = deserialize_message(blob, Image)
        ts = img.header.stamp.sec + img.header.stamp.nanosec * 1e-9
        arr = np.frombuffer(img.data, np.uint8).reshape(img.height, img.width, 3)
        bgr = arr[:, :, ::-1] if img.encoding == "rgb8" else arr   # rgb8 → bgr for cv2
        rel = f"rgb/{ts:.6f}.png"
        cv2.imwrite(str(args.out / rel), bgr)
        pairs.append((ts, rel))
    print(f"[ros2] 저장 {len(pairs)}장 {K['width']}x{K['height']}")

    intr = dict(fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"], width=K["width"], height=K["height"],
                depth_scale=1000.0,
                distortion=dict(model="inverse_brown_conrady",
                                k1=K["coeffs"][0], k2=K["coeffs"][1], p1=K["coeffs"][2],
                                p2=K["coeffs"][3], k3=K["coeffs"][4]),
                source=f"ros2:{Path(args.db3).name}", model="PINHOLE")
    (args.out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
    with open(args.out / "rgb.txt", "w") as f:
        f.write("# timestamp filename\n")
        for ts, rel in pairs:
            f.write(f"{ts:.6f} {rel}\n")
    print(f"[ros2] → {args.out}")


if __name__ == "__main__":
    main()
