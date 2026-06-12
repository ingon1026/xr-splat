#!/usr/bin/env bash
# run_colmap_sfm.sh — Phase 6 베이스라인: 키프레임 이미지에 COLMAP SfM → 포즈(임의 스케일).
#   register-all: hold-out 포함 전 키프레임을 등록(학습에서만 제외, 포즈 추정엔 포함).
#   WSLg GL 회피 위해 CPU SIFT. 산출: outputs/<scene>/colmap_sfm/0/{cameras,images,points3D}.bin
# usage: run_colmap_sfm.sh <scene>
set -eo pipefail
SCENE="${1:?usage: run_colmap_sfm.sh <scene>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROC="$ROOT/data/processed/$SCENE"
OUT="$ROOT/outputs/$SCENE/colmap_sfm"
COLMAP=/home/ingon/miniconda3/envs/colmap/bin/colmap
export LD_LIBRARY_PATH=/home/ingon/miniconda3/envs/colmap/lib
mkdir -p "$OUT"
DB="$OUT/database.db"; rm -f "$DB"
RGB="$(readlink -f "$PROC/rgb")"
# 키프레임 이름 리스트 (우리 COLMAP 모델의 등록 프레임 = ORB 키프레임)
grep -v '^#' "$PROC/colmap/sparse/0/images.txt" | awk 'NR%2==1{print $10}' > "$OUT/image_list.txt"
echo "[sfm] 키프레임 $(wc -l < "$OUT/image_list.txt")개 SfM 시작 $(date)"

# 고정 intrinsics = depth가 정합된 TUM intrinsics (통제 비교: 포즈 추정만 비교, intrinsics·depth 일관)
INTR=$(/home/ingon/miniconda3/envs/xrsplat/bin/python -c \
  "import json;d=json.load(open('$PROC/intrinsics.json'));print(f\"{d['fx']},{d['fy']},{d['cx']},{d['cy']}\")")
echo "[sfm] 고정 intrinsics: $INTR"
"$COLMAP" feature_extractor --database_path "$DB" --image_path "$RGB" \
  --image_list_path "$OUT/image_list.txt" \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$INTR" \
  --FeatureExtraction.use_gpu 0
"$COLMAP" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0
"$COLMAP" mapper --database_path "$DB" --image_path "$RGB" --output_path "$OUT" \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0

echo "[sfm] 등록 결과:"
"$COLMAP" model_analyzer --path "$OUT/0" 2>&1 | grep -iE "Cameras|Images|Points" | head
echo "[sfm] DONE $(date)"
