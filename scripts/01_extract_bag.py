#!/usr/bin/env python3
"""01_extract_bag.py — RGB-D 소스를 TUM RGB-D 포맷(processed/<scene>/)으로 추출·정규화.

모드 2개:
  bag : Intel RealSense .bag (오프라인, 카메라 불필요) → rs.align(depth→color) 후 추출. (D455 자체 캡처)
  dir : 기존 공개 데이터셋 디렉토리(TUM / Replica) 인제스트 → 동일 포맷으로 정규화. (M1 공개 데이터 진입)

출력(모드 공통) processed/<scene>/:
  rgb/<id>.<ext>       color
  depth/<id>.png       depth 16bit 1ch
  rgb.txt, depth.txt   "timestamp relpath"
  associations.txt     "rgb_ts rgb_path depth_ts depth_path"   (ORB-SLAM3 RGB-D 입력)
  intrinsics.json      이후 모든 단계의 단일 진실원

intrinsics.json 의 depth_scale 의미 (foot-gun 주의):
  depth_png_pixel_value / depth_scale = meters   (= ORB-SLAM3 RGBD.DepthMapFactor)
  RealSense SDK 의 depth_scale(meters/unit, 예 0.001)과 다른 값이다. 여기엔 DepthMapFactor를 저장한다.
  D455=1000, TUM=5000, Replica=6553.5.
"""
import argparse
import bisect
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("extract")

# TUM freiburg 카메라 프리셋 (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
# distortion(k1 k2 p1 p2 k3) = ORB-SLAM3 번들 TUMx.yaml 공식 계수
TUM_PRESETS = {
    "fr1": dict(fx=517.306408, fy=516.469215, cx=318.643040, cy=255.313989, width=640, height=480,
                distortion=dict(model="radtan", k1=0.262383, k2=-0.953104, p1=-0.005358, p2=0.002628, k3=1.163314)),
    "fr2": dict(fx=520.908620, fy=521.007327, cx=325.141442, cy=249.701764, width=640, height=480,
                distortion=dict(model="radtan", k1=0.231222, k2=-0.784899, p1=-0.003257, p2=-0.000105, k3=0.917205)),
    "fr3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, width=640, height=480,
                distortion=dict(model="radtan", k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)),
}
TUM_DEPTH_SCALE = 5000.0
REPLICA_DEPTH_SCALE = 6553.5  # cam_params.json 의 scale 가 있으면 그 값 우선
ZERO_DIST = dict(model="none", k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)
ASSOC_MAX_DIFF = 0.02  # rgb-depth 타임스탬프 매칭 허용오차 [s]


# ----------------------------------------------------------------------------- 공통 출력
def write_index_files(out: Path, pairs, intr: dict):
    """pairs: [(rgb_ts, rgb_rel, depth_ts, depth_rel)] — 이미지는 이미 out/ 아래 배치돼 있어야 함."""
    if not pairs:
        log.error("매칭된 rgb-depth 쌍이 0개 — 출력 중단")
        sys.exit(1)
    with open(out / "rgb.txt", "w") as fr, open(out / "depth.txt", "w") as fd, \
         open(out / "associations.txt", "w") as fa:
        fr.write("# timestamp filename\n")
        fd.write("# timestamp filename\n")
        for rts, rrel, dts, drel in pairs:
            fr.write(f"{rts:.6f} {rrel}\n")
            fd.write(f"{dts:.6f} {drel}\n")
            fa.write(f"{rts:.6f} {rrel} {dts:.6f} {drel}\n")
    intr = dict(intr)
    intr.setdefault("model", "PINHOLE")
    intr["_depth_scale_semantics"] = (
        "depth_png_pixel_value / depth_scale = meters (= ORB-SLAM3 RGBD.DepthMapFactor). "
        "RealSense SDK depth_scale(m/unit)와 혼동 금지."
    )
    with open(out / "intrinsics.json", "w") as f:
        json.dump(intr, f, indent=2, ensure_ascii=False)
    log.info("출력 완료: %d 쌍 → %s", len(pairs), out)


def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def associate(rgb, depth, max_diff=ASSOC_MAX_DIFF):
    """rgb/depth: [(ts, relpath)] 정렬 무관. 각 rgb에 최근접 depth 매칭(허용오차 내). TUM associate.py 로직."""
    depth_sorted = sorted(depth, key=lambda x: x[0])
    dts = [d[0] for d in depth_sorted]
    pairs, dropped = [], 0
    for ts, rrel in sorted(rgb, key=lambda x: x[0]):
        i = bisect.bisect_left(dts, ts)
        cand = [j for j in (i - 1, i) if 0 <= j < len(dts)]
        if not cand:
            dropped += 1
            continue
        best = min(cand, key=lambda j: abs(dts[j] - ts))
        if abs(dts[best] - ts) <= max_diff:
            pairs.append((ts, rrel, depth_sorted[best][0], depth_sorted[best][1]))
        else:
            dropped += 1
    if dropped:
        log.warning("매칭 실패로 버린 rgb 프레임: %d (총 rgb %d)", dropped, len(rgb))
    return pairs


# ----------------------------------------------------------------------------- bag 모드
def extract_bag(bag_path: Path, out: Path, depth_map_factor: float, max_frames: int):
    import numpy as np
    import cv2
    import pyrealsense2 as rs

    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)  # 배치 처리: 실시간 무시
    align = rs.align(rs.stream.color)

    depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
    ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cc = list(ci.coeffs)  # brown_conrady: [k1, k2, p1, p2, k3]
    dist = dict(model=str(ci.model).rsplit(".", 1)[-1], k1=cc[0], k2=cc[1], p1=cc[2], p2=cc[3], k3=cc[4])
    intr = dict(fx=ci.fx, fy=ci.fy, cx=ci.ppx, cy=ci.ppy, width=ci.width, height=ci.height,
                depth_scale=float(depth_map_factor), distortion=dist, source=f"bag:{bag_path.name}")
    log.info("color intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f %dx%d | SDK depth_scale=%.6g m/unit",
             ci.fx, ci.fy, ci.ppx, ci.ppy, ci.width, ci.height, depth_scale_m)

    pairs, n, last_fn = [], 0, -1
    while True:
        ok, frames = pipe.try_wait_for_frames(2000)
        if not ok:
            break
        fn = frames.get_frame_number()
        if fn <= last_fn:  # 재생 루프/되감김 감지 → 종료
            break
        last_fn = fn
        frames = align.process(frames)
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        if not d or not c:
            log.warning("프레임 %d: depth/color 누락 → skip", fn)
            continue
        ts = c.get_timestamp() / 1000.0  # ms → s
        color = np.asanyarray(c.get_data())
        if c.get_profile().format() == rs.format.rgb8:   # cv2.imwrite는 BGR 기대 → RGB면 변환
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        depth_raw = np.asanyarray(d.get_data()).astype(np.float32)
        depth_mm = np.clip(depth_raw * depth_scale_m * depth_map_factor, 0, 65535).astype(np.uint16)
        rgb_rel, depth_rel = f"rgb/{ts:.6f}.png", f"depth/{ts:.6f}.png"
        cv2.imwrite(str(out / rgb_rel), color)
        cv2.imwrite(str(out / depth_rel), depth_mm)
        pairs.append((ts, rgb_rel, ts, depth_rel))
        n += 1
        if max_frames and n >= max_frames:
            log.info("max-frames %d 도달 → 중단", max_frames)
            break
    pipe.stop()
    log.info("bag 추출: %d 프레임", n)
    write_index_files(out, pairs, intr)


# ----------------------------------------------------------------------------- dir 모드: TUM
def ingest_tum(src: Path, out: Path, variant: str):
    def read_list(p):
        items = []
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, rel = line.split()[0], line.split()[1]
            items.append((float(ts), rel))
        return items

    rgb = read_list(src / "rgb.txt")
    depth = read_list(src / "depth.txt")
    pairs = associate(rgb, depth)
    # 이미지 디렉토리 통째 심링크 (relpath 유지)
    symlink(src / "rgb", out / "rgb")
    symlink(src / "depth", out / "depth")
    intr = dict(TUM_PRESETS[variant], depth_scale=TUM_DEPTH_SCALE, source=f"tum:{variant}:{src.name}")
    write_index_files(out, pairs, intr)


# ----------------------------------------------------------------------------- dir 모드: Replica
def ingest_replica(src: Path, out: Path, cam_params: Path):
    cam = json.loads(Path(cam_params).read_text())["camera"]
    res = src / "results"
    rgbs = sorted(res.glob("frame*.jpg"))
    depths = sorted(res.glob("depth*.png"))
    if len(rgbs) != len(depths) or not rgbs:
        log.error("Replica results/ rgb(%d)/depth(%d) 불일치 또는 0개", len(rgbs), len(depths))
        sys.exit(1)
    pairs = []
    for i, (r, d) in enumerate(zip(rgbs, depths)):
        idx = r.stem.replace("frame", "")  # 000000
        rgb_rel, depth_rel = f"rgb/{idx}.jpg", f"depth/{idx}.png"
        symlink(r, out / rgb_rel)
        symlink(d, out / depth_rel)
        pairs.append((float(i), rgb_rel, float(i), depth_rel))  # 합성 timestamp = 프레임 인덱스
    intr = dict(fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
                width=cam["w"], height=cam["h"],
                depth_scale=float(cam.get("scale", REPLICA_DEPTH_SCALE)),
                distortion=ZERO_DIST, source=f"replica:{src.name}")
    write_index_files(out, pairs, intr)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("bag", help="RealSense .bag → TUM RGB-D")
    pb.add_argument("--bag", required=True, type=Path)
    pb.add_argument("--out", required=True, type=Path, help="processed/<scene>/")
    pb.add_argument("--depth-scale", type=float, default=1000.0, help="DepthMapFactor (D455 mm=1000)")
    pb.add_argument("--max-frames", type=int, default=0, help="0=전체")

    pd = sub.add_parser("dir", help="기존 TUM/Replica 디렉토리 인제스트")
    pd.add_argument("--dataset", required=True, choices=["tum", "replica"])
    pd.add_argument("--src", required=True, type=Path, help="시퀀스 디렉토리")
    pd.add_argument("--out", required=True, type=Path, help="processed/<scene>/")
    pd.add_argument("--variant", default="fr1", choices=list(TUM_PRESETS), help="TUM freiburg 프리셋")
    pd.add_argument("--cam-params", type=Path, help="Replica cam_params.json (기본: src/../cam_params.json)")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "bag":
        if not args.bag.exists():
            log.error("bag 없음: %s", args.bag); sys.exit(1)
        extract_bag(args.bag, args.out, args.depth_scale, args.max_frames)
    else:
        if not args.src.exists():
            log.error("src 없음: %s", args.src); sys.exit(1)
        if args.dataset == "tum":
            ingest_tum(args.src, args.out, args.variant)
        else:
            cp = args.cam_params or (args.src.parent / "cam_params.json")
            if not Path(cp).exists():
                log.error("Replica cam_params.json 없음: %s", cp); sys.exit(1)
            ingest_replica(args.src, args.out, cp)


if __name__ == "__main__":
    main()
