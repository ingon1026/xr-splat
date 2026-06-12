#!/usr/bin/env bash
# 06_train_gsplat.sh — gsplat 학습 (포즈 고정 + dense depth supervision). Phase 5.
#   입력 processed/<scene>/colmap/ → 산출 outputs/<scene>/gsplat/{scene.ply, train_log.jsonl, renders/}
#   프레임워크 택일: gsplat(+커스텀 depth loss) — 설치 사전검증·가벼움·COLMAP 직접·depth/pose 완전제어 (README 근거).
# usage: 06_train_gsplat.sh <scene> [iters]
set -eo pipefail
SCENE="${1:?usage: 06_train_gsplat.sh <scene> [iters]}"
ITERS="${2:-30000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /home/ingon/miniconda3/etc/profile.d/conda.sh
set +u; conda activate xrsplat; set -u
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="8.9" CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
echo "[06] gsplat 학습 scene=$SCENE iters=$ITERS  $(date)"
python "$ROOT/scripts/train_gsplat.py" --scene "$SCENE" --iters "$ITERS"
echo "[06] done → outputs/$SCENE/gsplat/scene.ply  $(date)"
