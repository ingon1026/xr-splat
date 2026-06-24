#!/usr/bin/env bash
# localize_query_colmap.sh — query 프레임을 orbframe 모델에 PnP 등록(=per-frame reloc, orbframe 좌표).
#   usage: localize_query_colmap.sh [n_query] [stride]
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COLMAP=/home/ingon/miniconda3/envs/colmap/bin/colmap
export LD_LIBRARY_PATH=/home/ingon/miniconda3/envs/colmap/lib
P=/home/ingon/miniconda3/envs/xrsplat/bin/python
NQ="${1:-80}"; STRIDE="${2:-4}"
REF="$ROOT/data/processed/ros2_bag2_home_rgbd_orbframe/colmap/sparse/0"
RGB="$ROOT/data/processed/ros2_bag2_home_rgbd/rgb"
OUT="$ROOT/outputs/ros2_bag2_home_rgbd/reloc_pnp"; mkdir -p "$OUT"; DB="$OUT/database.db"; rm -f "$DB"
INTR="642.284,641.448,641.204,366.335"   # fx,fy,cx,cy (orbframe intrinsics)

# reference KF 이름
grep -v '^#' "$REF/images.txt" | awk 'NF>=10 && NR%2==1{print $10}' | sort > "$OUT/ref_names.txt"
# reference 시간 범위 (orbframe은 녹화의 한 구간만 커버 — query는 그 안에서 골라야 겹침 확보)
REFMIN=$(sed 's/\.png$//' "$OUT/ref_names.txt" | sort -n | head -1)
REFMAX=$(sed 's/\.png$//' "$OUT/ref_names.txt" | sort -n | tail -1)
# query = ref 시간범위 안의 비-reference 프레임, STRIDE 간격 NQ개
ls "$RGB" | sed 's/\.png$//' | sort -n \
  | awk -v a="$REFMIN" -v b="$REFMAX" '$1>=a && $1<=b{print $1".png"}' \
  | grep -vxFf "$OUT/ref_names.txt" | awk "NR%$STRIDE==1" | head -n "$NQ" > "$OUT/query_names.txt"
# all = ref+query, 시간(숫자) 순 — sequential 매칭이 DB 삽입순=시간순을 따르도록
cat "$OUT/ref_names.txt" "$OUT/query_names.txt" | sed 's/\.png$//' | sort -n -u | sed 's/$/.png/' > "$OUT/all_names.txt"
echo "[loc] ref $(wc -l < "$OUT/ref_names.txt")  query $(wc -l < "$OUT/query_names.txt")  range $REFMIN~$REFMAX"

# 1) 특징 추출(ref+query) + sequential 매칭(시간순 인접 — exhaustive보다 훨씬 빠름)
"$COLMAP" feature_extractor --database_path "$DB" --image_path "$RGB" \
  --image_list_path "$OUT/all_names.txt" \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$INTR" --FeatureExtraction.use_gpu 0
"$COLMAP" sequential_matcher --database_path "$DB" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 30

# 2) reference 포즈 고정 시드(orbframe 좌표) → point_triangulator로 descriptor 모델
$P "$ROOT/scripts/make_orb_seed_from_db.py" --db "$DB" --orb "$REF" --out "$OUT/seed"
mkdir -p "$OUT/tri" "$OUT/registered"
"$COLMAP" point_triangulator --database_path "$DB" --image_path "$RGB" \
  --input_path "$OUT/seed" --output_path "$OUT/tri" --clear_points 1 \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0

# 3) query 프레임을 그 모델에 등록(포즈 추정) = per-frame reloc
"$COLMAP" image_registrator --database_path "$DB" --input_path "$OUT/tri" --output_path "$OUT/registered" \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
"$COLMAP" model_converter --input_path "$OUT/registered" --output_path "$OUT/registered" --output_type TXT
echo "[loc] registered 모델 → $OUT/registered ; query_names → $OUT/query_names.txt"
