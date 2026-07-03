#!/usr/bin/env python3
"""09_export_web.py — 3DGS .ply → .splat (antimatter15 포맷, 브라우저 뷰어용).

.splat = gaussian당 32바이트: pos 3×f32 · scale(exp) 3×f32 · RGBA 4×u8 · rot(quat→u8) 4×u8.
SH는 DC(f_dc_*)만 사용 — 웹 뷰어(gsplat.js)가 SH 미지원이므로 08의 lite 자산이 최적 입력.
importance(부피×불투명도) 내림차순 정렬 → 점진 로딩 시 큰 gaussian부터 보임.

CPU 전용(plyfile+numpy) — gsplat/GPU 불필요. 기존 스테이지(01~08) 무수정, 추가 전용.

usage:
  python scripts/09_export_web.py --scene replica_office0                  # lite 자산 → outputs/<scene>/web/<scene>.splat
  python scripts/09_export_web.py --scene <s> --target-mb 25               # 크기 상한(importance 상위만)
  python scripts/09_export_web.py --ply any.ply --out demo.splat --opacity-min 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SH_C0 = 0.28209479177387814
BYTES_PER_GAUSSIAN = 32


def load_gaussians(ply_path: Path):
    """3DGS ply → dict of numpy (raw 값 — activation 미적용)."""
    v = PlyData.read(str(ply_path))["vertex"].data
    names = v.dtype.names
    need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [n for n in need if n not in names]
    if missing:
        raise ValueError(f"3DGS ply 필드 없음: {missing} (in {ply_path})")
    g = {n: np.asarray(v[n], dtype=np.float32) for n in need}
    return g


def export_splat(g: dict, out_path: Path, opacity_min: float, max_gaussians: int):
    pos = np.stack([g["x"], g["y"], g["z"]], axis=1)                      # [N,3]
    scales = np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], axis=1))
    alpha = 1.0 / (1.0 + np.exp(-g["opacity"]))                           # sigmoid
    rgb = 0.5 + SH_C0 * np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], axis=1)
    quat = np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], axis=1)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)

    n_in = len(pos)

    # ── prune: 안 보이는 gaussian 제거 ──
    keep = alpha >= opacity_min
    pos, scales, alpha, rgb, quat = pos[keep], scales[keep], alpha[keep], rgb[keep], quat[keep]

    # ── importance 정렬(부피×불투명도) — 점진 로딩 품질 + 크기 캡 기준 ──
    importance = scales.prod(axis=1) * alpha
    order = np.argsort(-importance)
    if max_gaussians > 0:
        order = order[:max_gaussians]
    pos, scales, alpha, rgb, quat = pos[order], scales[order], alpha[order], rgb[order], quat[order]
    n_out = len(pos)

    # ── pack (32B/gaussian) ──
    rec = np.empty(n_out, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3),
                                 ("rgba", "u1", 4), ("rot", "u1", 4)])
    rec["pos"] = pos
    rec["scale"] = scales
    rec["rgba"][:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rec["rgba"][:, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rec["rot"] = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec.tofile(str(out_path))

    # ── 뷰어 초기 카메라 힌트 (index.html 상수에 박을 값) ──
    center = np.median(pos, axis=0)
    extent = float(np.max(pos.max(0) - pos.min(0)))
    meta = dict(n_input=int(n_in), n_output=int(n_out),
                size_mb=round(out_path.stat().st_size / 1e6, 2),
                center=[round(float(c), 3) for c in center],
                extent=round(extent, 2),
                suggested_radius=round(extent * 0.6, 2))
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="scene 이름 → 입력 outputs/<scene>/gsplat_<tag>/scene_lite.ply")
    ap.add_argument("--ply", type=Path, help="입력 ply 직접 지정 (--scene 무시)")
    ap.add_argument("--out", type=Path, help="출력 .splat 경로 (기본 outputs/<scene>/web/<scene>.splat)")
    ap.add_argument("--opacity-min", type=float, default=0.02, help="이하 alpha prune (기본 0.02)")
    ap.add_argument("--max-gaussians", type=int, default=0, help="importance 상위 N개만 (0=전체)")
    ap.add_argument("--target-mb", type=float, default=0, help="크기 상한 MB → max-gaussians 자동 산출")
    args = ap.parse_args()

    if args.ply:
        ply, out = args.ply, args.out or args.ply.with_suffix(".splat")
        scene = args.scene or ply.stem
    elif args.scene:
        from pipeline.config import load_config  # gsplat-free
        cfg = load_config(args.scene)
        ply = cfg.asset_lite
        out = args.out or (cfg.out_dir / "web" / f"{args.scene}.splat")
        scene = args.scene
    else:
        ap.error("--scene 또는 --ply 필요")

    max_g = args.max_gaussians
    if args.target_mb > 0:
        cap = int(args.target_mb * 1e6 / BYTES_PER_GAUSSIAN)
        max_g = min(max_g, cap) if max_g > 0 else cap

    print(f"[09] {ply}  →  {out}")
    g = load_gaussians(ply)
    meta = export_splat(g, out, args.opacity_min, max_g)

    est_10, est_50 = meta["size_mb"] * 8 / 10, meta["size_mb"] * 8 / 50
    print(f"[09] gaussians {meta['n_input']:,} → {meta['n_output']:,}  |  {meta['size_mb']} MB")
    print(f"[09] 예상 로딩: {est_10:.0f}s @10Mbps · {est_50:.0f}s @50Mbps")
    print(f"[09] 뷰어 카메라 힌트: target={meta['center']}  radius≈{meta['suggested_radius']} (extent {meta['extent']}m)")
    print(f"[09] meta → {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
