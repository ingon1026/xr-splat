#!/usr/bin/env python3
"""03_tum_to_colmap.py — KeyFrameTrajectory.txt(Twc) → COLMAP text 모델(Tcw). SPEC §3 Phase 3.

변환 로직은 pipeline/colmap_convert.py (단위테스트 tests/test_colmap_convert.py).
주의: cameras.txt는 PINHOLE(왜곡 미포함). 우리 이미지엔 왜곡이 있을 수 있으므로(TUM), Phase 5 학습과
특히 Phase 6 COLMAP-vs-ORB 베이스라인 비교에서 양쪽 경로가 동일한 왜곡 처리를 받아야 dB 비교가
오염되지 않는다. (PINHOLE은 SPEC 지정 — undistort/OPENCV 전환은 SPEC 변경 후 결정)

기하 정합성은 Phase 4 05_validate_poses.py("벽 한 겹") 에서 최종 검증된다 — 여기선 단위검증까지.

usage: 03_tum_to_colmap.py --scene <scene> [--tol 0.01]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.colmap_convert import (  # noqa: E402
    twc_to_tcw, load_tum, load_rgb_list, match_nearest,
    write_cameras_txt, write_images_txt, write_points3d_txt,
)

log = logging.getLogger("tum2colmap")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--traj", type=Path, help="기본: outputs/<scene>/slam/KeyFrameTrajectory.txt")
    ap.add_argument("--tol", type=float, default=0.01, help="키프레임 ts↔이미지 매칭 허용오차 [s]")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    proc = args.root / "data" / "processed" / args.scene
    traj = args.traj or (args.root / "outputs" / args.scene / "slam" / "KeyFrameTrajectory.txt")
    for p in (proc / "intrinsics.json", proc / "rgb.txt", traj):
        if not Path(p).exists():
            log.error("입력 없음: %s", p); sys.exit(1)

    intr = json.loads((proc / "intrinsics.json").read_text())
    rgb = sorted(load_rgb_list(proc / "rgb.txt"), key=lambda x: x[0])
    ts_keys = [r[0] for r in rgb]
    kfs = sorted(load_tum(traj), key=lambda k: k[0])

    out = proc / "colmap"
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    # 이미지 폴더 심링크 (COLMAP NAME = basename, images/<NAME>)
    img_link = out / "images"
    if img_link.is_symlink() or img_link.exists():
        img_link.unlink()
    img_link.symlink_to(proc / "rgb")   # 파이프라인 processed/<scene>/rgb 를 가리킴(raw 원본 직접 아님)

    entries, dropped = [], 0
    for ts, tx, ty, tz, qx, qy, qz, qw in kfs:
        name = match_nearest(ts, rgb, ts_keys, args.tol)
        if name is None:
            dropped += 1
            continue
        pose = twc_to_tcw(tx, ty, tz, qx, qy, qz, qw)   # Twc → Tcw (COLMAP order)
        entries.append((len(entries) + 1, pose, Path(name).name))
    if dropped:
        log.warning("이미지 매칭 실패 키프레임: %d (tol=%.3fs)", dropped, args.tol)
    if not entries:
        log.error("매칭된 키프레임 0 → 중단"); sys.exit(1)

    write_cameras_txt(sparse / "cameras.txt", intr["width"], intr["height"],
                      intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    write_images_txt(sparse / "images.txt", entries)
    write_points3d_txt(sparse / "points3D.txt")
    log.info("COLMAP 모델 작성: %d 이미지 (키프레임 %d) → %s", len(entries), len(kfs), sparse)


if __name__ == "__main__":
    main()
