#!/usr/bin/env python3
"""make_orbslam3_yaml.py — intrinsics.json → ORB-SLAM3 RGB-D 설정 yaml 자동생성.

검증된 ORB-SLAM3 번들 TUM yaml 구조를 그대로 재현하고, 카메라 파라미터/왜곡/DepthMapFactor를
intrinsics.json(단일 진실원)에서 치환한다. + System.SaveAtlasToFile(Atlas 맵).

주의(포맷): ORB-SLAM3 cv::FileStorage는 타입에 엄격하다. 실수는 반드시 소수점 포함(5000.0).
"""
import argparse
import json
from pathlib import Path


def fnum(x):
    """실수를 항상 소수점 포함 문자열로 (cv::FileStorage가 int/float 구분)."""
    return repr(float(x))


def make_yaml(intr: dict, atlas_name: str, nfeatures: int) -> str:
    d = intr.get("distortion", {})
    model = str(d.get("model", "none")).lower()
    # ORB-SLAM3 Camera1.k1..k3/p1/p2 는 forward Brown-Conrady(radtan) 계수를 기대한다.
    # RealSense color 스트림은 왜곡을 'inverse_brown_conrady'(역방향 모델)로 보고하는데, 이 계수를
    # radtan에 그대로 넣으면 매핑 방향이 어긋나 오히려 왜곡을 키운다. D455 color는 공장 정류로
    # 계수가 ~0이라, 역모델일 때는 0으로 두는 것이 정확하고 안전하다(직매핑 부정확 + 계수 무시 가능).
    # TUM 등 forward radtan/brown_conrady 계수는 그대로 사용. (SPEC §3 Phase 2)
    if model in ("inverse_brown_conrady", "inverse_modified_brown_conrady"):
        k1 = k2 = p1 = p2 = k3 = 0.0
        print(f"[yaml] distortion model='{model}' (역모델) → radtan 계수 0 처리")
    else:
        k1, k2, p1, p2, k3 = (d.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3"))
    w, h = int(intr["width"]), int(intr["height"])
    lines = [
        "%YAML:1.0", "",
        'File.version: "1.0"', "",
        'Camera.type: "PinHole"', "",
        f"Camera1.fx: {fnum(intr['fx'])}",
        f"Camera1.fy: {fnum(intr['fy'])}",
        f"Camera1.cx: {fnum(intr['cx'])}",
        f"Camera1.cy: {fnum(intr['cy'])}", "",
        f"Camera1.k1: {fnum(k1)}",
        f"Camera1.k2: {fnum(k2)}",
        f"Camera1.p1: {fnum(p1)}",
        f"Camera1.p2: {fnum(p2)}",
        f"Camera1.k3: {fnum(k3)}", "",
        f"Camera.width: {w}",
        f"Camera.height: {h}", "",
        "Camera.fps: 30",
        "Camera.RGB: 1", "",
        "Stereo.ThDepth: 40.0",
        "Stereo.b: 0.07732",  # TUM 튜닝값. D455/Replica 깊은 장면에선 재검토 필요
        "",
        f"RGBD.DepthMapFactor: {fnum(intr['depth_scale'])}", "",
        f"ORBextractor.nFeatures: {int(nfeatures)}",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7", "",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8",
        "Viewer.ViewpointF: 500.0", "",
    ]
    if atlas_name:
        lines += [f'System.SaveAtlasToFile: "{atlas_name}"', ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intrinsics", required=True, type=Path, help="processed/<scene>/intrinsics.json")
    ap.add_argument("--out", required=True, type=Path, help="출력 yaml 경로")
    ap.add_argument("--atlas", default="", help="System.SaveAtlasToFile 이름 (예: 장면명). 빈값이면 미저장")
    ap.add_argument("--nfeatures", type=int, default=1250, help="ORBextractor.nFeatures (트래킹 로스 시 2000)")
    args = ap.parse_args()
    intr = json.loads(args.intrinsics.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(make_yaml(intr, args.atlas, args.nfeatures))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
