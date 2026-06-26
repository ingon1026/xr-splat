#!/usr/bin/env bash
# 06_train_gsplat.sh — gsplat 학습 (포즈 고정 + dense depth supervision). Phase 5.
#   입력 processed/<scene>/colmap/ → 산출 outputs/<scene>/gsplat/{scene.ply, train_log.jsonl, renders/}
#   프레임워크 택일: gsplat(+커스텀 depth loss) — 설치 사전검증·가벼움·COLMAP 직접·depth/pose 완전제어 (README 근거).
# usage: 06_train_gsplat.sh <scene> [iters]
set -eo pipefail
SCENE="${1:?usage: 06_train_gsplat.sh <scene> [iters]}"
ITERS="${2:-30000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
echo "[06] gsplat 학습 scene=$SCENE iters=$ITERS  $(date)"
"$XRSPLAT_PYTHON" "$ROOT/scripts/train_gsplat.py" --scene "$SCENE" --iters "$ITERS"
echo "[06] done → outputs/$SCENE/gsplat/scene.ply  $(date)"
