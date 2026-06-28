#!/usr/bin/env bash
# run_home_orbframe.sh — home orbframe(=snap 수정본) 05 게이트 → gsplat 재학습.
# LEGACY: 단일 scene 빌드는 `python xrsplat.py build configs/<scene>.yaml`로 통합됨 (게이트·resume·결과팩 자동).
#   전제: snap_scene_to_orb.py로 ros2_bag2_home_rgbd_orbframe 생성 완료(ORB 프레임 정합 확인됨).
#   §2 게이트: 05 PASS 전 학습 금지 → FAIL이면 학습 진입 없이 종료.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
SCENE=ros2_bag2_home_rgbd_orbframe

echo "===== KeyFrameTrajectory 생성 (colmap → TUM Twc) $(date) ====="
mkdir -p "$ROOT/outputs/$SCENE/slam"
python - "$ROOT" "$SCENE" <<'PY'
import sys; root,scene=sys.argv[1],sys.argv[2]; sys.path.insert(0,root)
import numpy as np
from pipeline.backproject import read_colmap_images, colmap_world_RT
from scipy.spatial.transform import Rotation as Rot
imgs=read_colmap_images(f'{root}/data/processed/{scene}/colmap/sparse/0/images.txt')
with open(f'{root}/outputs/{scene}/slam/KeyFrameTrajectory.txt','w') as f:
    for n in sorted(imgs,key=lambda n:float(n.rsplit('.',1)[0])):
        R_wc,C=colmap_world_RT(*imgs[n]); q=Rot.from_matrix(np.asarray(R_wc)).as_quat()
        ts=n.rsplit('.',1)[0]
        f.write(f"{ts} {C[0]:.7f} {C[1]:.7f} {C[2]:.7f} {q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n")
print(f"  KF {len(imgs)} → outputs/{scene}/slam/KeyFrameTrajectory.txt")
PY

echo "===== 05 게이트 (large D455 scene 보정: baseline 0.15-0.4, pass-cm 4.0) $(date) ====="
# 보정 근거: home median depth 1.8~2.7m → D455 노이즈 바닥 3.4~7cm, 룸스케일 3.0cm은 센서 바닥 아래.
#   baseline 축소=드리프트 누적↓(05 line70), pass-cm 4.0=관측 잔차(2.7~3.3cm) 위 + gross오류(10cm+) 아래.
# 측정은 COLMAP images.txt(Tcw)로 하므로(마스크만 KeyFrameTrajectory) → 음성대조는 images.txt를 섭동해야 함.
GATE_ARGS="--baseline 0.15 0.4 --pass-cm 4.0"
IMG="$ROOT/data/processed/$SCENE/colmap/sparse/0/images.txt"

echo "--- 음성대조: images.txt를 회전3°+TZ10cm 섭동 → 같은 임계에서 FAIL 해야 게이트 살아있음 ---"
cp "$IMG" "$IMG.real"
python "$ROOT/scripts/_perturb_images.py" --in "$IMG.real" --out "$IMG" --rot-deg 3 --tz-m 0.10
if python "$ROOT/scripts/05_validate_poses.py" --scene "$SCENE" $GATE_ARGS >/dev/null 2>&1; then
  echo "[home] ⚠ 음성대조 섭동이 PASS — 게이트가 고무도장(임계 과대). 중단."; cp "$IMG.real" "$IMG"; rm -f "$IMG.real"; exit 3
else
  echo "[home] 음성대조 OK: 섭동 images.txt는 FAIL (게이트 살아있음)"
fi
cp "$IMG.real" "$IMG"; rm -f "$IMG.real"

echo "--- 실제 게이트 ---"
if python "$ROOT/scripts/05_validate_poses.py" --scene "$SCENE" $GATE_ARGS; then
  echo "[home] 05 PASS ✅ (scene-scale 보정 + 음성대조 통과) → 재학습 진입"
else
  echo "[home] 05 FAIL ❌ (rc=$?) — 보정 임계에서도 실패면 진짜 포즈 문제. 학습 중단."; exit 2
fi

echo "===== gsplat 재학습 (포즈 고정+depth, holdout 8) $(date) ====="
python "$ROOT/scripts/train_gsplat.py" --scene "$SCENE" --iters 30000 --holdout-every 8

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
print(f"  비교> 기존 sfmsnap(프레임 깨짐) held-out 24.34 / room2 22.38")
PY
echo "===== DONE $(date)  → outputs/$SCENE/gsplat/scene.ply ====="
