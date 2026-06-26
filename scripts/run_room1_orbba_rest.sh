#!/usr/bin/env bash
# run_room1_orbba_rest.sh — Strategy A room1 후속(②~⑤): finalize → 05 게이트 → gsplat → prune.
#   ① run_orb_seeded_ba.sh d455_room1 가 ba_final 생성 완료된 뒤 실행.
#   §2 게이트: 05_validate_poses.py PASS 전 학습 금지 → 05 FAIL이면 학습 진입 없이 종료.
# ponytail: room1 전용 1회 드라이버. 다른 scene 일반화는 필요해지면 그때.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
P=python

BA="$ROOT/outputs/d455_room1/orb_seeded_ba/ba_final"
EXCL="$ROOT/outputs/d455_room1/gsplat_exclude.txt"
SCENE=d455_room1_orbba

[ -f "$BA/images.txt" ] || { echo "[rest] BA 미완성: $BA/images.txt 없음 — ① 먼저"; exit 1; }

echo "===== ② finalize (ORB metric 스냅 + 씬 구성) $(date) ====="
$P "$ROOT/scripts/finalize_orbba_scene.py" --scene d455_room1 --ba-model "$BA"

echo "===== ③ 05 게이트 (룸스케일 baseline) $(date) ====="
if $P "$ROOT/scripts/05_validate_poses.py" --scene "$SCENE" --baseline 0.1 0.28 --overlap-min 0.15; then
  echo "[rest] 05 PASS ✅ → 학습 진입"
else
  echo "[rest] 05 FAIL/CANNOT ❌ (rc=$?) — §2 게이트: 학습 중단. 포즈 점검 필요."; exit 2
fi

echo "===== ④ gsplat 학습 (포즈 고정 + depth, 사람 170KF 제외) $(date) ====="
$P "$ROOT/scripts/train_gsplat.py" --scene "$SCENE" --iters 30000 --exclude-list "$EXCL"

echo "===== ⑤ prune floaters $(date) ====="
$P "$ROOT/scripts/prune_floaters.py" --scene "$SCENE" \
  --model "$ROOT/outputs/$SCENE/gsplat/scene.ply" \
  --out   "$ROOT/outputs/$SCENE/gsplat/scene_pruned.ply"

echo "===== DONE $(date)  → outputs/$SCENE/gsplat/ ====="
