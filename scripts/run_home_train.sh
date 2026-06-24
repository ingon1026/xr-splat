#!/usr/bin/env bash
# run_home_train.sh — home orbframe 학습만(05 게이트는 이미 PASS·음성대조 통과 검증됨, 생략).
#   n_gauss 2M까지 성장 후 refine-stop(15000) plateau는 정상(기존 sfmsnap 동일). OOM 아님.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /home/ingon/miniconda3/etc/profile.d/conda.sh; set +u; conda activate xrsplat; set -u
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="8.9" CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
SCENE=ros2_bag2_home_rgbd_orbframe

echo "===== gsplat 재학습 (포즈 고정+depth, holdout 8, refine-stop 5000) $(date) ====="
# orbframe은 default refine-stop(15000)이면 densification 폭주(step6700 5.7M, OOM직전). sfmsnap은
# step~10000에 2.0M로 정지(입증된 수). orbframe은 step5000에 이미 2.07M이므로 refine-stop 5000으로
# 같은 ~2M에서 densification 정지 → OOM 회피 + 이후 25k step은 순수 최적화. (M1도 hold-out 발산에 early refine-stop)
python "$ROOT/scripts/train_gsplat.py" --scene "$SCENE" --iters 30000 --holdout-every 8 --refine-stop 5000

echo "===== holdout PSNR 평가 $(date) ====="
python - "$ROOT" "$SCENE" <<'PY'
import sys; root,scene=sys.argv[1],sys.argv[2]; sys.path.insert(0,root)
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation as Rot
from pipeline.gsplat_io import load_ply, render
from pipeline.backproject import read_colmap_cameras, read_colmap_images
DEV='cuda'; proc=f'{root}/data/processed/{scene}'; gdir=f'{root}/outputs/{scene}/gsplat'
W,H,fx,fy,cx,cy=read_colmap_cameras(f'{proc}/colmap/sparse/0/cameras.txt')
K=torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1.]],device=DEV)
imgs=read_colmap_images(f'{proc}/colmap/sparse/0/images.txt')
g=load_ply(f'{gdir}/scene.ply',DEV)
ho=[l.strip() for l in open(f'{gdir}/holdout.txt') if l.strip()]
ps=[]
for n in ho:
    if n not in imgs: continue
    qw,qx,qy,qz,tx,ty,tz=imgs[n]; vm=np.eye(4,dtype=np.float32)
    vm[:3,:3]=Rot.from_quat([qx,qy,qz,qw]).as_matrix(); vm[:3,3]=[tx,ty,tz]
    pr=render(g,torch.tensor(vm,device=DEV),K,W,H).detach().cpu().numpy()
    gt=cv2.imread(f'{proc}/rgb/{n}')[:,:,::-1].astype(np.float32)/255.
    ps.append(float(-10*np.log10(max(((pr-gt)**2).mean(),1e-10))))
a=np.array(ps)
print(f"[home orbframe] holdout {len(ps)}뷰  PSNR median {np.median(a):.2f} mean {a.mean():.2f} min {a.min():.2f}")
print(f"  비교> 기존 sfmsnap(프레임 깨짐) 24.34 / room2(프레임 OK) 22.38")
PY
echo "===== DONE $(date) ====="
