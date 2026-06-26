#!/usr/bin/env bash
# build_train70_scene.sh — 70% 가우시안 학습용 processed scene 조립.
#   colmap = reloc7030/tri (ref 70% KF 포즈 + 그 위 삼각측량 점)  → 가우시안은 ref만 본다(자동 보장).
#   rgb/depth = 원본 심링크, associations/intrinsics = 원본 복사.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/data/processed/ros2_bag2_home_rgbd_orbframe"
TRI="$ROOT/outputs/ros2_bag2_home_rgbd_orbframe/reloc7030/tri"
DST="$ROOT/data/processed/ros2_bag2_home_rgbd_orbframe_train70"

[ -f "$TRI/images.txt" ] || { echo "FATAL: $TRI/images.txt 없음(build_7030_reloc 먼저)"; exit 1; }
mkdir -p "$DST/colmap/sparse/0"
cp "$TRI/cameras.txt" "$TRI/images.txt" "$TRI/points3D.txt" "$DST/colmap/sparse/0/"
# rgb/depth 심링크(전체 — train_gsplat는 images.txt에 있는 이름만 읽음)
ln -sfn "$SRC/rgb" "$DST/rgb"
ln -sfn "$SRC/depth" "$DST/depth"
cp "$SRC/associations.txt" "$SRC/intrinsics.json" "$DST/"

NIMG=$(grep -v '^#' "$DST/colmap/sparse/0/images.txt" | awk 'NF>=10 && NR%2==1' | wc -l)
NPTS=$(grep -vc '^#' "$DST/colmap/sparse/0/points3D.txt")
echo "[train70] scene → $DST  (images $NIMG, points3D $NPTS)"
