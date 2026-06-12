#!/usr/bin/env bash
# 02_run_orbslam3.sh — processed/<scene>/ 에 ORB-SLAM3 RGB-D를 headless로 실행.
#   intrinsics.json → yaml 자동생성(DepthMapFactor 일치 + distortion + Atlas) → rgbd_tum(viewer OFF).
#   산출물: outputs/<scene>/slam/{KeyFrameTrajectory.txt, <scene>.osa}
# 성공 판정은 종료코드가 아니라 KeyFrameTrajectory.txt 생성으로 한다 (CLAUDE.md §4).
# usage: 02_run_orbslam3.sh <scene> [nfeatures]
#   트래킹 로스로 완주 실패 시 nfeatures를 올려 재시도 (예: 1250 → 2000) — SPEC Phase 2.
set -eo pipefail
SCENE="${1:?usage: 02_run_orbslam3.sh <scene> [nfeatures]}"
NFEAT="${2:-1250}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROC="$ROOT/data/processed/$SCENE"
OUT="$ROOT/outputs/$SCENE/slam"
CFG="$ROOT/configs/${SCENE}_orbslam3.yaml"
ORB="$ROOT/third_party/ORB_SLAM3"
VOC="$ORB/Vocabulary/ORBvoc.txt"
BIN="$ORB/Examples/RGB-D/rgbd_tum"   # headless (0002 viewer-off 패치)

[ -f "$PROC/intrinsics.json" ]   || { echo "[02] no intrinsics.json: $PROC"; exit 1; }
[ -f "$PROC/associations.txt" ]  || { echo "[02] no associations.txt: $PROC"; exit 1; }
[ -x "$BIN" ] && [ -f "$VOC" ]   || { echo "[02] ORB-SLAM3 빌드/Vocabulary 없음 (third_party/build_orbslam3.sh)"; exit 1; }

mkdir -p "$OUT"
# 1) intrinsics.json → ORB-SLAM3 yaml (DepthMapFactor 반드시 intrinsics.json과 일치)
python3 "$ROOT/pipeline/make_orbslam3_yaml.py" \
    --intrinsics "$PROC/intrinsics.json" --out "$CFG" --atlas "$SCENE" --nfeatures "$NFEAT"

# 2) headless 실행 (KeyFrameTrajectory.txt, <scene>.osa 는 CWD 기준 → OUT에서 실행)
cd "$OUT"
export DISPLAY="${DISPLAY:-:0}"
echo "[02] ORB-SLAM3 RGB-D (headless) scene=$SCENE nfeatures=$NFEAT  $(date)"
"$BIN" "$VOC" "$CFG" "$PROC" "$PROC/associations.txt" || echo "[02] (binary exit $? — 판정은 출력 파일로)"

# 3) 판정: KeyFrameTrajectory.txt 생성 여부 (CLAUDE.md §4)
echo "[02] 산출물:"; ls -la "$OUT"/KeyFrameTrajectory.txt "$OUT/${SCENE}.osa" 2>/dev/null || true
[ -s "$OUT/KeyFrameTrajectory.txt" ] && echo "[02] OK: KeyFrameTrajectory ($(wc -l < "$OUT/KeyFrameTrajectory.txt") KF)" \
    || { echo "[02] FAIL: KeyFrameTrajectory.txt 없음/빈 파일"; exit 1; }
