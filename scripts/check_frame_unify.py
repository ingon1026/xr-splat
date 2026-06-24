#!/usr/bin/env python3
"""check_frame_unify.py — 두 COLMAP 모델의 카메라 중심 사이 Sim3(Umeyama) 계산·리포트.

용도 2가지:
  (G1 게이트, Strategy A) BA정제 포즈 vs 원본 ORB 포즈 → scale≈1·회전≈0 이면 "ORB 좌표계 보존"(통합 성공).
      per-pose 잔차(~cm)는 정상 = 교정량 자체. 판별자는 scale/rotation.
  (B 폴백) ORB 포즈 → 가우시안(SfM) 포즈 정렬용 Sim3를 sim3.json으로 영속화.

Umeyama 구현은 scripts/07_evaluate.py 와 동일. 이미지 NAME 으로 매칭.
usage: check_frame_unify.py --ref <ref images.txt> --new <new images.txt> [--out sim3.json] [--expect-identity]
       (Sim3는 new → ref 방향: C_ref ≈ s*R@C_new + t)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.backproject import read_colmap_images, colmap_world_RT  # noqa: E402


def umeyama(src, dst):  # 동일 구현: scripts/07_evaluate.py
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Sc.T @ Dc / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / ((Sc ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def centers(images_txt):
    imgs = read_colmap_images(images_txt)
    return {n: colmap_world_RT(*p)[1] for n, p in imgs.items()}  # t_wc = 카메라 중심


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="기준 모델 images.txt (예: 원본 ORB)")
    ap.add_argument("--new", required=True, type=Path, help="정렬 대상 images.txt (예: BA정제 / SfM)")
    ap.add_argument("--out", type=Path, help="Sim3 저장 경로(json)")
    ap.add_argument("--expect-identity", action="store_true", help="G1: scale≈1·회전≈0 PASS 판정")
    args = ap.parse_args()

    cr, cn = centers(args.ref), centers(args.new)
    common = sorted(set(cr) & set(cn))
    if len(common) < 3:
        print(f"[unify] 공통 이미지 {len(common)}개 — 너무 적음"); sys.exit(1)
    Cref = np.array([cr[n] for n in common])
    Cnew = np.array([cn[n] for n in common])

    s, R, t = umeyama(Cnew, Cref)                       # new → ref
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    aligned = (s * (R @ Cnew.T).T + t)
    resid = np.sqrt(((aligned - Cref) ** 2).sum(1))     # Sim3 정렬 후 잔차
    raw = np.sqrt(((Cnew - Cref) ** 2).sum(1))          # 정렬 전 raw per-pose 변위(=교정량, A에서 ~cm 기대)

    print(f"[unify] 공통 {len(common)}개  scale={s:.5f}  rot={ang:.3f}°  |t|={np.linalg.norm(t)*100:.2f}cm")
    print(f"[unify] Sim3정렬후 잔차  median={np.median(resid)*100:.2f}cm  max={resid.max()*100:.2f}cm")
    print(f"[unify] raw per-pose 변위 median={np.median(raw)*100:.2f}cm  max={raw.max()*100:.2f}cm")

    if args.out:
        args.out.write_text(json.dumps(dict(
            scale=s, R=R.tolist(), t=t.tolist(), n=len(common),
            resid_median_cm=float(np.median(resid) * 100), rot_deg=ang), indent=2))
        print(f"[unify] → {args.out}")

    if args.expect_identity:
        ok = (0.99 <= s <= 1.01) and ang < 2.0
        print(f"[unify] G1 {'PASS' if ok else 'FAIL'} (scale∈[0.99,1.01] & rot<2°): scale={s:.5f} rot={ang:.3f}°")
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
