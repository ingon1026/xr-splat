#!/usr/bin/env python3
"""test_build_report.py — verdict 결정 트리 단위 검증 (gsplat 불필요·순수 로직).

run: python scripts/test_build_report.py   (또는 pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _report(ply_mb=473.0, frame="pass", sr=1.0, pose="present", quality="present"):
    return {
        "asset": {"scene_ply_mb": ply_mb},
        "frame_integrity": {"status": frame},
        "relocalization": {"success_rate": sr, "status": "present" if sr is not None else "missing"},
        "pose_validation": {"status": pose},
        "quality": {"status": quality},
    }


def test_verdict_tree():
    # verdict는 numpy/colmap import 없이 호출 가능해야 함(함수만 import)
    from build_report import verdict

    # 자산 없음 → INCOMPLETE (프레임 깨져도 자산부재 우선)
    assert verdict(_report(ply_mb=None, frame="fail"))[0] == "INCOMPLETE"
    # 프레임 깨짐 → FRAME_INVALID (reloc 100%·PSNR 무관)
    assert verdict(_report(frame="fail", sr=1.0))[0] == "FRAME_INVALID"
    # 프레임 미검증 → RENDER_ONLY
    assert verdict(_report(frame="missing", sr=1.0))[0] == "RENDER_ONLY"
    # 프레임 무결 + reloc<0.5 → NEEDS_RECAPTURE
    assert verdict(_report(frame="pass", sr=0.3))[0] == "NEEDS_RECAPTURE"
    # 프레임 무결 + reloc 0.5~0.8 → RENDER_ONLY (부분 재진입)
    assert verdict(_report(frame="pass", sr=0.65))[0] == "RENDER_ONLY"
    # 프레임 무결 + reloc>=0.8 → XR_READY
    assert verdict(_report(frame="pass", sr=0.9))[0] == "XR_READY"
    # 프레임 무결 + reloc 없음 → RENDER_ONLY (재진입 미입증)
    assert verdict(_report(frame="pass", sr=None))[0] == "RENDER_ONLY"
    print("test_verdict_tree: OK (7 cases)")


if __name__ == "__main__":
    test_verdict_tree()
