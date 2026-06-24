#!/usr/bin/env python3
"""make_orb_seed_from_db.py — COLMAP database에서 직접 4.x 시드 모델 생성(mapper 불필요).

mapper는 sequential 매칭에서 조각나기 쉬워(긴 시퀀스) 일부 이미지만 등록 → 골격 불완전.
database는 모든 이미지/rig/frame을 이미 갖고 있으므로, 거기서 rigs/frames/images.txt를 직접 쓰고
포즈만 ORB로 채운다. point_triangulator는 이 시드 + db 매칭으로 전 이미지 삼각측량.

usage: make_orb_seed_from_db.py --db <database.db> --orb <ORB 03 model dir> --out <seed dir>
"""
import argparse, sqlite3, shutil
from pathlib import Path

SENSOR = {0: "CAMERA", 1: "IMU"}


def read_orb_poses(orb_images_txt):
    out = {}
    for l in open(orb_images_txt):
        if l.startswith("#"):
            continue
        s = l.split()
        if len(s) >= 10:                 # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            out[s[9]] = " ".join(s[1:8])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--orb", required=True, type=Path, help="ORB 03 모델 dir (poses + cameras.txt)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    orb = read_orb_poses(args.orb / "images.txt")
    shutil.copy(args.orb / "cameras.txt", args.out / "cameras.txt")     # 동일 PINHOLE 카메라
    (args.out / "points3D.txt").write_text("# 3D point list (비움 — triangulator 재생성)\n")

    con = sqlite3.connect(str(args.db)); cur = con.cursor()
    id2name = dict(cur.execute("SELECT image_id, name FROM images"))
    rigs = cur.execute("SELECT rig_id, ref_sensor_id, ref_sensor_type FROM rigs").fetchall()
    frames = cur.execute("SELECT frame_id, rig_id FROM frames ORDER BY frame_id").fetchall()
    fdata = {}
    for fid, did, sid, stype in cur.execute("SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data"):
        fdata.setdefault(fid, []).append((stype, sid, did))

    # rigs.txt: RIG_ID NUM_SENSORS REF_SENSOR_TYPE REF_SENSOR_ID (단일 카메라 rig)
    with open(args.out / "rigs.txt", "w") as f:
        f.write("# RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, ...\n")
        for rid, ref_sid, ref_stype in rigs:
            f.write(f"{rid} 1 {SENSOR.get(ref_stype, 'CAMERA')} {ref_sid}\n")

    # images.txt: IMAGE_ID <ORB pose> CAMERA_ID NAME + 빈 2D줄
    miss = 0
    with open(args.out / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (ORB 시드)\n")
        for iid, name in sorted(id2name.items()):
            pose = orb.get(name)
            if pose is None:
                miss += 1; continue
            f.write(f"{iid} {pose} 1 {name}\n\n")

    # frames.txt: FRAME_ID RIG_ID <ORB pose> NUM_DATA [SENSOR_TYPE SENSOR_ID DATA_ID]...
    with open(args.out / "frames.txt", "w") as f:
        f.write("# FRAME_ID RIG_ID RIG_FROM_WORLD(qw qx qy qz tx ty tz) NUM_DATA DATA_IDS (ORB 시드)\n")
        for fid, rid in frames:
            data = fdata.get(fid, [])
            ref = next((d for d in data if d[0] == 0), data[0] if data else None)  # 카메라 측정
            if ref is None:
                continue
            name = id2name.get(ref[2]); pose = orb.get(name)
            if pose is None:
                continue
            ids = " ".join(f"{SENSOR.get(st,'CAMERA')} {si} {di}" for st, si, di in data)
            f.write(f"{fid} {rid} {pose} {len(data)} {ids}\n")

    print(f"[seed-db] → {args.out}  이미지 {len(id2name)} (포즈누락 {miss}), 프레임 {len(frames)}, rig {len(rigs)}")


if __name__ == "__main__":
    main()
