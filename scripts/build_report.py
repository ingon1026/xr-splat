#!/usr/bin/env python3
"""build_report.py — collect existing scene artifacts and emit an XR-readiness report.

This is intentionally lightweight: it does not render, train, or run COLMAP. It only reads
artifacts that already exist on disk and refuses to infer frame integrity from PSNR alone.

Example:
  python scripts/build_report.py --scene ros2_bag2_home_rgbd_orbframe --tag mcmc2m --frame-ok
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.backproject import read_colmap_images, colmap_world_RT  # noqa: E402
from check_frame_unify import umeyama  # noqa: E402  (Sim3, 재사용)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def compute_frame_integrity(asset_images_txt, orb_traj, eps_s=0.005):
    """자산 COLMAP 포즈 vs ORB TUM 궤적을 타임스탬프로 매칭 → Sim3(scale/rot).
    scale≈1·rot≈0 이면 자산이 ORB 좌표계 보존(프레임 무결) = 런타임 정합 가능.
    PSNR로는 못 잡는 게이트 — 깨진 프레임은 PSNR 높아도 런타임 무용."""
    if not asset_images_txt.exists() or not orb_traj.exists():
        return {"status": "missing"}
    asset = read_colmap_images(asset_images_txt)
    orb_lines = [l.split() for l in open(orb_traj) if l.strip() and not l.startswith("#")]
    orb_ts = np.array([float(s[0]) for s in orb_lines])
    orb_c = {float(s[0]): np.array([float(s[1]), float(s[2]), float(s[3])]) for s in orb_lines}
    Cnew, Cref = [], []
    for n, p in asset.items():
        try:
            a = float(Path(n).stem)
        except ValueError:
            continue
        j = int(np.abs(orb_ts - a).argmin())
        if abs(orb_ts[j] - a) < eps_s:
            Cnew.append(colmap_world_RT(*p)[1]); Cref.append(orb_c[orb_ts[j]])
    if len(Cnew) < 10:
        return {"status": "missing", "matched": len(Cnew), "note": "공통 KF<10 — ORB 궤적 매칭 실패"}
    Cnew, Cref = np.array(Cnew), np.array(Cref)
    s, R, t = umeyama(Cnew, Cref)
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    ok = (0.99 <= s <= 1.01) and rot < 2.0
    return {"status": "pass" if ok else "fail", "scale": round(float(s), 5),
            "rot_deg": round(rot, 3), "matched": len(Cnew)}


def count_lines(path):
    if not path.exists():
        return None
    return sum(1 for line in open(path) if line.strip() and not line.startswith("#"))


def read_colmap_names(images_txt):
    if not images_txt.exists():
        return []
    names = []
    for line in open(images_txt):
        s = line.split()
        if len(s) >= 10 and not line.startswith("#"):
            names.append(s[9])
    return names


def size_mb(path):
    return round(path.stat().st_size / (1024 * 1024), 1) if path.exists() else None


def tail_jsonl(path):
    if not path.exists():
        return None
    last = None
    for line in open(path):
        line = line.strip()
        if line:
            last = line
    return json.loads(last) if last else None


def asset_dir(root, scene, tag):
    return root / "outputs" / scene / ("gsplat" + (f"_{tag}" if tag else ""))


def collect_reloc(root, scene, reloc_dir):
    rd = reloc_dir or (root / "outputs" / scene / "reloc7030")
    query = rd / "query_names.txt"
    registered = rd / "registered" / "images.txt"
    qnames = [l.strip() for l in open(query)] if query.exists() else []
    rnames = set(read_colmap_names(registered))
    ok = len([q for q in qnames if q in rnames])
    return {
        "dir": str(rd),
        "query_count": len(qnames) if qnames else None,
        "registered_query_count": ok if qnames else None,
        "success_rate": round(ok / len(qnames), 4) if qnames else None,
        "status": "present" if qnames and registered.exists() else "missing",
    }


def verdict(report):
    missing = []
    if report["asset"]["scene_ply_mb"] is None:
        missing.append("scene.ply")
    if report["pose_validation"]["status"] != "present":
        missing.append("05_validate artifacts")
    if report["frame_integrity"]["status"] != "pass":
        missing.append("frame integrity pass")
    if report["quality"]["status"] == "missing":
        missing.append("quality/postprocess metrics")
    if report["relocalization"]["status"] == "missing":
        missing.append("relocalization evidence")

    # 1) 자산 없음
    if report["asset"]["scene_ply_mb"] is None:
        return "INCOMPLETE", missing
    # 2) 프레임 깨짐 → PSNR 무관하게 런타임 무용 (이 게이트가 build_report 존재 이유)
    if report["frame_integrity"]["status"] == "fail":
        return "FRAME_INVALID", missing
    # 3) 프레임 미검증(ORB 기준 없음) → 렌더는 되나 런타임 정합 보장 못 함
    if report["frame_integrity"]["status"] != "pass":
        return "RENDER_ONLY", missing
    # 4) 프레임 무결 → reloc(재진입) 능력으로 등급
    sr = report["relocalization"]["success_rate"]
    if sr is None:
        return "RENDER_ONLY", missing          # 렌더 가능, 재진입 미입증
    if sr < 0.5:
        return "NEEDS_RECAPTURE", missing       # 자기 프레임도 등록 못 함 = 캡처 부적합
    if sr >= 0.8:
        return "XR_READY", missing              # 프레임 무결 + 재진입 검증
    return "RENDER_ONLY", missing               # 부분 재진입


def write_markdown(path, report):
    lines = [
        f"# xr-splat Asset Report: {report['scene']} ({report['tag'] or 'default'})",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        "## Summary",
        "",
        f"- Frames: rgb={report['inputs']['rgb_count']} keyframes={report['inputs']['keyframe_count']} colmap_images={report['inputs']['colmap_image_count']}",
        f"- Asset: scene.ply={report['asset']['scene_ply_mb']} MB, scene_lite.ply={report['asset']['scene_lite_ply_mb']} MB",
        f"- Pose validation: {report['pose_validation']['status']} ({report['pose_validation']['overlay_count']} overlay files)",
        f"- Frame integrity: **{report['frame_integrity']['status']}** "
        + (f"(auto Sim3: scale={report['frame_integrity']['auto']['scale']}, rot={report['frame_integrity']['auto']['rot_deg']}°, "
           f"{report['frame_integrity']['auto']['matched']} matched KF)" if report['frame_integrity'].get('auto') and report['frame_integrity']['auto'].get('status') != 'missing'
           else f"(method: {report['frame_integrity']['method']})"),
        f"- Relocalization: {report['relocalization']['success_rate']} success rate "
        f"({report['relocalization']['registered_query_count']}/{report['relocalization']['query_count']} queries)",
        f"- Coverage: novel-view render PSNR={report['coverage']['novel_render_psnr']} "
        f"(novelty median {report['coverage']['novelty_cm_median']}cm)" if report['coverage']['status'] == 'present'
        else "- Coverage: (no novel-view eval)",
        "",
        "## Missing Or Blocking Evidence",
        "",
    ]
    if report["missing"]:
        lines += [f"- {m}" for m in report["missing"]]
    else:
        lines.append("- none")
    lines += [
        "",
        "## Notes",
        "",
        "- This report does not render or train. It only summarizes existing artifacts.",
        "- PSNR/SSIM/LPIPS are not sufficient for XR readiness without frame integrity.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="", help="gsplat tag, e.g. mcmc2m -> outputs/<scene>/gsplat_mcmc2m")
    ap.add_argument("--sim3-json", type=Path, help="Optional Sim3/frame integrity JSON")
    ap.add_argument("--frame-ok", action="store_true", help="Mark frame integrity as externally verified")
    ap.add_argument("--orb-traj", type=Path,
                    help="ORB KeyFrameTrajectory.txt for AUTO frame-integrity Sim3 "
                         "(default: outputs/<scene-'_orbframe'>/slam/KeyFrameTrajectory.txt)")
    ap.add_argument("--reloc-dir", type=Path, help="Optional reloc evidence dir (default: outputs/<scene>/reloc7030)")
    args = ap.parse_args()

    root, scene = args.root, args.scene
    proc = root / "data" / "processed" / scene
    out_scene = root / "outputs" / scene
    gdir = asset_dir(root, scene, args.tag)
    sparse = proc / "colmap" / "sparse" / "0"
    sim3 = read_json(args.sim3_json) if args.sim3_json else None
    post = read_json(gdir / "postprocess.json")

    # ORB 기준 궤적 자동 유도(scene에서 '_orbframe' 제거) — 자동 프레임무결성용
    orb_traj = args.orb_traj
    if orb_traj is None:
        orb_scene = scene[:-len("_orbframe")] if scene.endswith("_orbframe") else scene
        orb_traj = root / "outputs" / orb_scene / "slam" / "KeyFrameTrajectory.txt"

    integrity = None
    if args.frame_ok:
        frame_status = "pass"
    elif (integrity := compute_frame_integrity(sparse / "images.txt", orb_traj))["status"] != "missing":
        frame_status = integrity["status"]                       # 자동 Sim3 계산 결과
    elif sim3:                                                    # 폴백: 외부 sim3.json
        scale = float(sim3.get("scale", 0)); rot = float(sim3.get("rot_deg", 999))
        frame_status = "pass" if 0.99 <= scale <= 1.01 and rot < 2.0 else "fail"
    else:
        frame_status = "missing"

    # 커버리지/새시점 신호(있으면) — 율속이 캡처 커버리지임을 드러내는 envelope
    novel = read_json(gdir / "novel_7030_eval.json")

    report = {
        "scene": scene,
        "tag": args.tag,
        "inputs": {
            "rgb_count": count_lines(proc / "rgb.txt"),
            "keyframe_count": count_lines(out_scene / "slam" / "KeyFrameTrajectory.txt"),
            "colmap_image_count": len(read_colmap_names(sparse / "images.txt")) if (sparse / "images.txt").exists() else None,
        },
        "asset": {
            "dir": str(gdir),
            "scene_ply_mb": size_mb(gdir / "scene.ply"),
            "scene_lite_ply_mb": size_mb(gdir / "scene_lite.ply"),
            "holdout_count": count_lines(gdir / "holdout.txt"),
            "last_train_log": tail_jsonl(gdir / "train_log.jsonl"),
        },
        "pose_validation": {
            "status": "present" if (out_scene / "validate").exists() else "missing",
            "overlay_count": len(list((out_scene / "validate").glob("*.ply"))) if (out_scene / "validate").exists() else 0,
        },
        "frame_integrity": {
            "status": frame_status,
            "method": "frame-ok-flag" if args.frame_ok else ("auto-sim3" if integrity and integrity["status"] != "missing" else ("sim3-json" if sim3 else "none")),
            "auto": integrity,                                   # 자동 계산(scale/rot/matched)
            "orb_traj": str(orb_traj) if integrity and integrity.get("status") != "missing" else None,
            "sim3_json": str(args.sim3_json) if args.sim3_json else None,
        },
        "quality": {
            "status": "present" if post else "missing",
            "postprocess": post,
        },
        "relocalization": collect_reloc(root, scene, args.reloc_dir),
        "coverage": {
            "status": "present" if novel else "missing",
            "novelty_cm_median": novel["novelty_cm"]["median"] if novel else None,
            "novel_render_psnr": novel["psnr"]["coarse"] if novel else None,
            "note": "novel-view(외삽) render PSNR — 낮으면 캡처 커버리지 한계" if novel else None,
        },
    }
    report["verdict"], report["missing"] = verdict(report)

    report_path = out_scene / f"asset_report_{args.tag or 'default'}.json"
    md_path = out_scene / f"asset_report_{args.tag or 'default'}.md"
    out_scene.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_markdown(md_path, report)

    print(f"[report] {report['verdict']} -> {report_path}")
    print(f"[report] markdown -> {md_path}")


if __name__ == "__main__":
    main()
