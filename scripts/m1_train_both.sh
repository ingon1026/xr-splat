#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/ingon/slam/xr-splat
source /home/ingon/miniconda3/etc/profile.d/conda.sh
set +u; conda activate xrsplat; set -u
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.9 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
echo "=== M1 (a) ORB-pose 학습 holdout8 15k $(date) ==="
python $ROOT/scripts/train_gsplat.py --scene tum_fr1_desk --holdout-every 8 --tag m1 --iters 15000 --refine-stop 7000
echo "=== M1 (b) COLMAP-pose 학습 holdout8 15k $(date) ==="
python $ROOT/scripts/train_gsplat.py --scene tum_fr1_desk_sfm --holdout-every 8 --tag m1 --iters 15000 --refine-stop 7000
echo "=== M1 양쪽 학습 완료 $(date) ==="
