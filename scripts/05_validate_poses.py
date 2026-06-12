#!/usr/bin/env python3
"""05_validate_poses.py — 포즈 기하 정합 게이트 ("벽 한 겹"). **05 PASS 전 gsplat 학습 금지** (CLAUDE.md §2).

설계 (advisor):
- 페어 선정·겹침 마스크는 **신뢰 Twc**(KeyFrameTrajectory)로. 검증 대상 COLMAP Tcw로 마스크를 잡으면
  진짜 ghost가 게이트 밖으로 빠져 false-PASS 되기 때문(마스크는 포즈 버그와 독립이어야 함).
- 정합 **측정은 COLMAP Tcw 역투영**으로 — 이게 gsplat이 쓰는 포즈이고, 03 CLI 배선(포즈↔이미지↔depth
  매칭)을 검증한다. (Tcw=inv(Twc)는 이미 단위테스트로 증명됨; 05는 수학이 아니라 배선/depth 일관성 검증)
- TUM은 PINHOLE+렌즈왜곡이라 full-frame이 왜곡에 지배됨(2m서 코너 ~10cm) → **중앙 크롭이 1차 포즈 판정**,
  full-frame은 왜곡 진단용.
- 판정: 겹침 영역 최근접거리 **percentile(p50)** < 임계. 평균은 비겹침 점에 오염됨.
- 페어 못 찾으면 **"cannot validate"** (절대 default-PASS 금지).
- 05가 blind: 균일 depth_scale 오차(둘 다 같이 스케일→ghost 없음). 그건 Phase1 Z-range가 커버.

FAIL 트리아지 순서: 왜곡(중앙크롭 통과여부) → 배선(포즈↔이미지↔depth) → depth 정합.

검증/한계 (negative control로 확인):
- 포즈에 TZ/회전 섭동을 주입하면 05가 FAIL → 나쁜 포즈 탐지 작동(항상-PASS 아님).
- **NN 거리는 in-plane 평행 이동(지배 평면을 따라 미끄러짐)에 둔감.** gross 역변환은 평면 밖이라 잡히고,
  off-by-one 배선 버그는 association으로 별도 배제(kf_ts==rgb_ts, ‖C−t_wc‖≈0).
- knobs(--baseline/--pass-cm/--crop)는 **TUM 책상 스케일 기본값** — Replica/D455 룸스케일은 재검증 필요(Stereo.b와 같은 footgun).
- 큰 baseline에서 cm급 잔차는 실제 SLAM 드리프트(M1 ORB-vs-COLMAP·Phase6 ATE가 정량화). 05는 fine-accuracy 체크가 아님.

usage: 05_validate_poses.py --scene <scene>
"""
import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.colmap_convert import load_tum, load_rgb_list, match_nearest  # noqa: E402
from pipeline.backproject import (  # noqa: E402
    read_colmap_images, tum_world_RT, colmap_world_RT, backproject, nn_distances,
)
from scipy.spatial import cKDTree  # noqa: E402

log = logging.getLogger("validate")


def load_record(proc, name, twc, tcw, depth_rel, fx, fy, cx, cy, ds, stride):
    """같은 픽셀 집합에 대해 Twc/Tcw 월드점을 둘 다 구한다(인덱스 정렬). 반환 None이면 skip."""
    dpath = proc / depth_rel
    if not dpath.exists():
        return None
    depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED).astype(np.float32) / ds
    R_wt, t_wt = tum_world_RT(*twc)
    R_wc, t_wc = colmap_world_RT(*tcw)
    w_twc, pix, _ = backproject(depth, fx, fy, cx, cy, R_wt, t_wt, stride=stride)
    w_tcw, _, _ = backproject(depth, fx, fy, cx, cy, R_wc, t_wc, stride=stride)
    return dict(name=name, twc_world=w_twc, tcw_world=w_tcw, pix=pix, center=t_wt)


def pct(d, q):
    return float(np.percentile(d, q)) if len(d) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--baseline", type=float, nargs=2, default=[0.2, 0.6],
                    help="페어 baseline 범위 [m]. moderate가 적정: 너무 작으면 약한 게이트, 너무 크면 SLAM 드리프트가 3cm 마진 잠식")
    ap.add_argument("--overlap-gate", type=float, default=0.05, help="겹침/마스크 게이트 [m] (신뢰 Twc 기준)")
    ap.add_argument("--overlap-min", type=float, default=0.2, help="co-visible 최소 겹침 비율")
    ap.add_argument("--crop", type=float, default=0.6, help="중앙 크롭 비율(1차 판정)")
    ap.add_argument("--pass-cm", type=float, default=3.0, help="PASS 임계 (percentile 거리)")
    ap.add_argument("--pairs", type=int, default=3, help="검증할 페어 수")
    ap.add_argument("--ts-min", type=float, default=float("-inf"),
                    help="이 타임스탬프 미만 키프레임 제외 (동적 구간 배제용)")
    ap.add_argument("--ts-max", type=float, default=float("inf"),
                    help="이 타임스탬프 초과 키프레임 제외 (동적 구간 배제용)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    proc = args.root / "data" / "processed" / args.scene
    sparse = proc / "colmap" / "sparse" / "0"
    intr = json.loads((proc / "intrinsics.json").read_text())
    fx, fy, cx, cy, ds = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["depth_scale"]
    W, H = int(intr["width"]), int(intr["height"])
    imgs = read_colmap_images(sparse / "images.txt")
    assoc = {Path(p[1]).name: p[3] for p in (l.split() for l in open(proc / "associations.txt")) if len(p) >= 4}
    rgb = sorted(load_rgb_list(proc / "rgb.txt"), key=lambda x: x[0])
    ts_keys = [r[0] for r in rgb]

    # 키프레임 레코드 (Twc + Tcw 동일 픽셀)
    kf_path = args.root / "outputs" / args.scene / "slam" / "KeyFrameTrajectory.txt"
    recs = []
    for ts, tx, ty, tz, qx, qy, qz, qw in sorted(load_tum(kf_path), key=lambda k: k[0]):
        if ts < args.ts_min or ts > args.ts_max:   # 동적(사람) 구간 키프레임 배제
            continue
        rel = match_nearest(ts, rgb, ts_keys, args.tol)
        if rel is None:
            continue
        name = Path(rel).name
        if name not in imgs or name not in assoc:
            continue
        r = load_record(proc, name, (tx, ty, tz, qx, qy, qz, qw), imgs[name], assoc[name],
                        fx, fy, cx, cy, ds, args.stride)
        if r:
            recs.append(r)
    log.info("키프레임 레코드 %d개", len(recs))
    if len(recs) < 2:
        log.error("레코드 부족 → cannot validate"); sys.exit(2)

    # 페어 선정: 신뢰 Twc 클라우드로 baseline + 겹침
    trees_twc = [cKDTree(r["twc_world"]) for r in recs]
    cands = []
    for i, j in combinations(range(len(recs)), 2):
        bl = np.linalg.norm(recs[i]["center"] - recs[j]["center"])
        if not (args.baseline[0] <= bl <= args.baseline[1]):
            continue
        d = trees_twc[j].query(recs[i]["twc_world"], k=1)[0]
        ov = float(np.mean(d <= args.overlap_gate))
        if ov >= args.overlap_min:
            cands.append((ov, bl, i, j))
    # 강한 게이트: co-visible 중 baseline 큰 쪽 우선(시점차 클수록 ghost·왜곡이 드러남), 동률이면 overlap
    cands.sort(key=lambda c: (round(c[1], 2), c[0]), reverse=True)
    if not cands:
        log.error("co-visible + baseline 페어 없음 → CANNOT VALIDATE (default-PASS 금지)"); sys.exit(2)

    outdir = args.root / "outputs" / args.scene / "validate"
    outdir.mkdir(parents=True, exist_ok=True)
    import open3d as o3d

    results = []
    for k, (ov, bl, i, j) in enumerate(cands[:args.pairs]):
        A, B = recs[i], recs[j]
        # 겹침 마스크 = A의 Twc점이 B의 Twc클라우드 게이트 내 (포즈버그 독립)
        dA = trees_twc[j].query(A["twc_world"], k=1)[0]
        m_full = dA <= args.overlap_gate
        u, v = A["pix"][:, 0], A["pix"][:, 1]
        cw, ch = args.crop, args.crop
        m_center = m_full & (u >= W * (1 - cw) / 2) & (u <= W * (1 + cw) / 2) & \
                            (v >= H * (1 - ch) / 2) & (v <= H * (1 + ch) / 2)
        # 측정 = COLMAP Tcw 역투영점의 최근접거리 (마스크는 Twc로 고정)
        d_full = nn_distances(A["tcw_world"][m_full], B["tcw_world"]) if m_full.any() else np.array([])
        d_cen = nn_distances(A["tcw_world"][m_center], B["tcw_world"]) if m_center.any() else np.array([])
        center_p50 = pct(d_cen, 50)
        verdict = (m_center.sum() > 50) and (center_p50 <= args.pass_cm / 100.0)
        results.append(dict(A=A["name"], B=B["name"], ov=ov, bl=bl,
                            center=dict(n=int(m_center.sum()), p50=center_p50, p90=pct(d_cen, 90)),
                            full=dict(n=int(m_full.sum()), p50=pct(d_full, 50), p90=pct(d_full, 90)),
                            pass_=bool(verdict)))
        # depth 오버레이(.ply): A(Tcw, 빨강) + B(Tcw, 초록)
        pc = o3d.geometry.PointCloud()
        pa, pb = A["tcw_world"][m_full], B["tcw_world"]
        pc.points = o3d.utility.Vector3dVector(np.vstack([pa, pb]))
        pc.colors = o3d.utility.Vector3dVector(np.vstack([np.tile([1, 0, 0], (len(pa), 1)),
                                                          np.tile([0, 1, 0], (len(pb), 1))]))
        o3d.io.write_point_cloud(str(outdir / f"pair{k}_{A['name']}__{B['name']}.ply"), pc)

    print("\n===== 05 VALIDATE (중앙크롭 1차 판정 / full-frame=왜곡진단) =====")
    for r in results:
        print(f"[{'PASS' if r['pass_'] else 'FAIL'}] {r['A']} vs {r['B']} | overlap={r['ov']:.2f} baseline={r['bl']:.2f}m")
        print(f"     center(n={r['center']['n']}): p50={r['center']['p50']*100:.1f}cm p90={r['center']['p90']*100:.1f}cm  ← 1차")
        print(f"     full  (n={r['full']['n']}): p50={r['full']['p50']*100:.1f}cm p90={r['full']['p90']*100:.1f}cm  (왜곡 포함)")
    allpass = all(r["pass_"] for r in results)
    print(f"\n=> {'PASS ✅ (포즈 기하 정합)' if allpass else 'FAIL ❌'}  [pairs={len(results)}, 임계 p50<{args.pass_cm}cm]")
    if not allpass:
        print("   트리아지: 중앙크롭도 FAIL이면 배선(포즈↔이미지↔depth)/depth 의심. "
              "중앙 PASS·full FAIL이면 왜곡(가장자리). 오버레이 .ply로 ghost 방향 확인:", outdir)
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
