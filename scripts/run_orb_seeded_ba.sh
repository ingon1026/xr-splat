#!/usr/bin/env bash
# run_orb_seeded_ba.sh — Strategy A 전체 파이프라인: ORB 포즈 시드 → COLMAP BA 정제.
#   재투영 BA는 전역 scale/rot/trans가 null-space → ORB metric 좌표계 보존 + cm급 per-pose 교정(=+5dB).
#   COLMAP 4.x는 rigs/frames.txt를 요구 → db에서 직접 골격 생성 + ORB 포즈 주입(make_orb_seed_from_db.py).
#   전제: 03_tum_to_colmap.py가 ORB COLMAP 모델(data/processed/<scene>/colmap/sparse/0) 생성 완료.
# usage: run_orb_seeded_ba.sh <scene> [matcher: seq|exhaustive] [overlap]
set -eo pipefail
SCENE="${1:?usage: run_orb_seeded_ba.sh <scene> [seq|exhaustive] [overlap]}"
MATCHER="${2:-seq}"          # 연속 핸드헬드 시퀀스=seq(빠름), 정적 비연속=exhaustive
OVERLAP="${3:-20}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROC="$ROOT/data/processed/$SCENE"
ORBMODEL="$PROC/colmap/sparse/0"
OUT="$ROOT/outputs/$SCENE/orb_seeded_ba"
COLMAP=/home/ingon/miniconda3/envs/colmap/bin/colmap
export LD_LIBRARY_PATH=/home/ingon/miniconda3/envs/colmap/lib
P=/home/ingon/miniconda3/envs/xrsplat/bin/python
mkdir -p "$OUT"; DB="$OUT/database.db"; rm -f "$DB"
RGB="$(readlink -f "$PROC/rgb")"
[ -f "$ORBMODEL/images.txt" ] || { echo "[orbba] ORB 모델 없음: $ORBMODEL (먼저 03 실행)"; exit 1; }

grep -v '^#' "$ORBMODEL/images.txt" | awk 'NF>=10 && NR%2==1{print $10}' > "$OUT/image_list.txt"
NKF=$(wc -l < "$OUT/image_list.txt")
INTR=$($P -c "import json;d=json.load(open('$PROC/intrinsics.json'));print(f\"{d['fx']},{d['fy']},{d['cx']},{d['cy']}\")")
echo "[orbba] $SCENE  KF $NKF  intrinsics $INTR  matcher=$MATCHER  $(date)"

# 1) 특징 추출 + 매칭
"$COLMAP" feature_extractor --database_path "$DB" --image_path "$RGB" \
  --image_list_path "$OUT/image_list.txt" \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$INTR" --FeatureExtraction.use_gpu 0
if [ "$MATCHER" = exhaustive ]; then
  "$COLMAP" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0
else
  "$COLMAP" sequential_matcher --database_path "$DB" --FeatureMatching.use_gpu 0 \
    --SequentialMatching.overlap "$OVERLAP"
fi

# 2-3) db에서 직접 4.x 시드 골격 생성(rigs/frames/images.txt) + ORB 포즈 주입.
#   mapper 경유는 sequential 매칭에서 조각나 일부만 등록 → db-직접이 전 이미지 보장(분열 없음).
$P "$ROOT/scripts/make_orb_seed_from_db.py" --db "$DB" --orb "$ORBMODEL" --out "$OUT/seed"

# 4) point_triangulator(포즈 고정) → bundle_adjuster(포즈+점 정제) 1라운드 (2라운드는 발산)
mkdir -p "$OUT/tri_seed" "$OUT/ba_final"
"$COLMAP" point_triangulator --database_path "$DB" --image_path "$RGB" \
  --input_path "$OUT/seed" --output_path "$OUT/tri_seed" --clear_points 1 \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
"$COLMAP" bundle_adjuster --input_path "$OUT/tri_seed" --output_path "$OUT/ba_final" \
  --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 0
"$COLMAP" model_converter --input_path "$OUT/ba_final" --output_path "$OUT/ba_final" --output_type TXT
echo "FINAL_MODEL=$OUT/ba_final" > "$OUT/final_model.txt"
echo "[orbba] 최종 BA 모델:"; "$COLMAP" model_analyzer --path "$OUT/ba_final" 2>&1 | grep -iE "Images|Points|reproj" | head
echo "[orbba] DONE $(date)  → finalize_orbba_scene.py 로 ORB 스냅+씬 구성"
