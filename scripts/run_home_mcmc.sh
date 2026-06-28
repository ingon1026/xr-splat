#!/usr/bin/env bash
# run_home_mcmc.sh — [home 정식 학습 드라이버: MCMC strategy 채택, 배포 자산 mcmc2m 생성]
# LEGACY: 단일 scene 빌드는 `python xrsplat.py build configs/<scene>.yaml`로 통합됨 (게이트·resume·결과팩 자동).
# home orbframe을 MCMC strategy로 재학습(cap_max 스윕).
#   판별 실험: 2M(현 default와 동수)·3M(더 높은 안정 cap)을 같은 인자로 학습해
#   "soft"가 배치 문제인지 raw-count 천장인지 가른다. 평가는 07_evaluate.py(SSIM/LPIPS).
#   MCMC는 cap_max로 하드캡 → DefaultStrategy의 transient 5.7M OOM 없음.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
SCENE=ros2_bag2_home_rgbd_orbframe

# 학습 인자는 현 default 자산과 동일 고정(apples-to-apples). 변수는 strategy/cap_max뿐.
for CAP in 2000000 3000000; do
  TAG="mcmc$((CAP/1000000))m"
  echo "===== MCMC 학습 cap_max=$CAP tag=$TAG $(date) ====="
  python "$ROOT/scripts/train_gsplat.py" --scene "$SCENE" --iters 30000 --holdout-every 8 \
    --strategy mcmc --cap-max "$CAP" --tag "$TAG"
done
echo "===== 학습 DONE $(date). 다음: 07_evaluate.py로 default/mcmc2m/mcmc3m SSIM·LPIPS 비교 ====="
