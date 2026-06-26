#!/usr/bin/env bash
# build_7030_reloc.sh — 70/30 진짜-새-시점 reloc 파이프라인 (localize_query_colmap.sh 차용).
#   ref = 앞 70% KF(시간순), query = 뒤 30% KF.  → 순환성 깨기:
#   - seed(고정포즈)는 **ref 70%만** 으로 만든다(ref-only orb 디렉터리). query KF 포즈가 새지 않도록.
#   - 매칭은 exhaustive(query 꼬리가 ref와 공간적으로 겹치는 loop를 sequential은 못 잡음).
#   출력: outputs/ros2_bag2_home_rgbd_orbframe/reloc7030/{tri,registered,ref_names.txt,query_names.txt,ref_orb/}
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COLMAP=/home/ingon/miniconda3/envs/colmap/bin/colmap
export LD_LIBRARY_PATH=/home/ingon/miniconda3/envs/colmap/lib
P=/home/ingon/miniconda3/envs/xrsplat/bin/python
PCT="${1:-70}"   # ref 비율(%)
REF_MODEL="$ROOT/data/processed/ros2_bag2_home_rgbd_orbframe/colmap/sparse/0"
RGB="$ROOT/data/processed/ros2_bag2_home_rgbd_orbframe/rgb"
OUT="$ROOT/outputs/ros2_bag2_home_rgbd_orbframe/reloc7030"; mkdir -p "$OUT"
DB="$OUT/database.db"; rm -f "$DB"
INTR="642.284,641.448,641.204,366.335"   # fx,fy,cx,cy (orbframe intrinsics)

# 전체 KF 이름(시간순=타임스탬프 숫자순)
grep -v '^#' "$REF_MODEL/images.txt" | awk 'NF>=10 && NR%2==1{print $10}' | sed 's/\.png$//' | sort -n | sed 's/$/.png/' > "$OUT/all_kf.txt"
TOTAL=$(wc -l < "$OUT/all_kf.txt")
NREF=$(( TOTAL * PCT / 100 ))
head -n "$NREF" "$OUT/all_kf.txt" > "$OUT/ref_names.txt"
tail -n +"$((NREF+1))" "$OUT/all_kf.txt" > "$OUT/query_names.txt"
cat "$OUT/ref_names.txt" "$OUT/query_names.txt" | sed 's/\.png$//' | sort -n -u | sed 's/$/.png/' > "$OUT/all_names.txt"
echo "[7030] total $TOTAL  ref(${PCT}%) $(wc -l < "$OUT/ref_names.txt")  query $(wc -l < "$OUT/query_names.txt")"
echo "[7030] ref 시간범위 $(head -1 "$OUT/ref_names.txt" | sed 's/.png//') ~ $(tail -1 "$OUT/ref_names.txt" | sed 's/.png//') ; query $(head -1 "$OUT/query_names.txt" | sed 's/.png//') ~ $(tail -1 "$OUT/query_names.txt" | sed 's/.png//')"

# ref-only orb 디렉터리: cameras.txt 복사 + images.txt엔 ref KF 포즈줄만 (query 포즈 누출 차단)
REFORB="$OUT/ref_orb"; mkdir -p "$REFORB"
cp "$REF_MODEL/cameras.txt" "$REFORB/cameras.txt"
{ echo "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (ref 70% only)"
  grep -v '^#' "$REF_MODEL/images.txt" | awk 'NF>=10 && NR%2==1' | grep -wFf "$OUT/ref_names.txt"
} > "$REFORB/images.txt"
echo "# 3D point list (비움)" > "$REFORB/points3D.txt"
NSEED_POSE=$(grep -v '^#' "$REFORB/images.txt" | awk 'NF>=10' | wc -l)
echo "[7030] ref_orb images.txt 포즈줄 $NSEED_POSE (== ref $(wc -l < "$OUT/ref_names.txt") 이어야)"

# 1) 특징 추출(ref+query 전부) + exhaustive 매칭(query 꼬리-ref loop 포착)
"$COLMAP" feature_extractor --database_path "$DB" --image_path "$RGB" \
  --image_list_path "$OUT/all_names.txt" \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$INTR" --FeatureExtraction.use_gpu 0
"$COLMAP" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0

# 2) ref-only 고정포즈 시드 → point_triangulator (ref descriptor 모델)
$P "$ROOT/scripts/make_orb_seed_from_db.py" --db "$DB" --orb "$REFORB" --out "$OUT/seed"
# seed sanity: query 이름이 seed/images.txt에 들어가면 안 됨
NLEAK=$(grep -v '^#' "$OUT/seed/images.txt" | awk 'NF>=10{print $NF}' | grep -wFf "$OUT/query_names.txt" | wc -l || true)
echo "[7030] seed query 누출 $NLEAK (0 이어야)"
[ "$NLEAK" -ne 0 ] && { echo "[7030] FATAL: query 포즈가 seed로 누출됨"; exit 1; }
mkdir -p "$OUT/tri" "$OUT/registered"
"$COLMAP" point_triangulator --database_path "$DB" --image_path "$RGB" \
  --input_path "$OUT/seed" --output_path "$OUT/tri" --clear_points 1 \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
"$COLMAP" model_converter --input_path "$OUT/tri" --output_path "$OUT/tri" --output_type TXT

# 3) query 30% KF를 ref 모델에 등록(reloc)
"$COLMAP" image_registrator --database_path "$DB" --input_path "$OUT/tri" --output_path "$OUT/registered" \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
"$COLMAP" model_converter --input_path "$OUT/registered" --output_path "$OUT/registered" --output_type TXT

# 자가 게이트: 30% query 중 등록 성공률
REGED=$(grep -v '^#' "$OUT/registered/images.txt" | awk 'NF>=10{print $NF}' | grep -wFf "$OUT/query_names.txt" | sort -u | wc -l)
NQ=$(wc -l < "$OUT/query_names.txt")
echo "[7030] ===== query 등록 성공률: $REGED / $NQ ====="
echo "[7030] registered 모델 → $OUT/registered"
