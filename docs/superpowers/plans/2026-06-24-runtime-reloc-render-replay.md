# Runtime reloc→render replay 데모 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 저장된 home ORB 맵에 프레임을 relocalize하고 그 pose로 home 가우시안 맵을 렌더해 '실제 프레임 | 가우시안 렌더' 리플레이 영상을 만들어, decoupled의 런타임 절반(reloc→render)을 증명한다.

**Architecture:** 2단계 파일 결합. Stage 1: ORB-SLAM3 RGB-D를 localization mode(고정 맵)로 home 프레임 리플레이 → 프레임별 pose+상태 로그. Stage 2: 그 pose로 orbframe 가우시안 렌더 + 실제 rgb와 side-by-side → 영상. reloc pose와 가우시안 맵이 동일 ORB 프레임이라 변환 없이 결합.

**Tech Stack:** ORB-SLAM3(C++, RGB-D), gsplat(python), pipeline.gsplat_io, imageio, conda env `xrsplat`(CUDA 12.1).

## Global Constraints

- ORB-SLAM3는 **항상 뷰어 OFF(headless)** — `System(...,RGBD,false)`. WSLg Pangolin segfault 회피. 성공 판정은 종료코드 아닌 출력 파일.
- 5분 이상 작업은 **nohup 백그라운드 + 로그**. `pkill -f`에 실행 명령줄과 겹치는 문자열 금지(자기 셸 kill).
- gsplat 학습/렌더 env: `conda activate xrsplat` + `export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.9 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11` (gsplat JIT가 gcc-11 필요).
- `render(g, viewmat, K, W, H)`의 viewmat은 **Tcw 4x4**(world→camera). TUM Twc(C, R_wc)→Tcw: `R_cw=R_wc.T; vm[:3,:3]=R_cw; vm[:3,3]=-R_cw@C`.
- 좌표: reloc pose(`ros2_bag2_home_rgbd.osa` ORB 프레임) == 가우시안 자산(`ros2_bag2_home_rgbd_orbframe`, 그 ORB 런에 snap) 프레임. 변환 없이 직결.
- 커밋: Conventional Commits, AI 귀속 트레일러 금지, author=ingon1026.

**경로 상수:**
- home rgb: `data/processed/ros2_bag2_home_rgbd/rgb/`, depth: `.../depth/`, associations: `.../associations.txt`, intrinsics: `.../intrinsics.json`
- home 맵: `outputs/ros2_bag2_home_rgbd/slam/ros2_bag2_home_rgbd.osa`
- 가우시안 자산: `outputs/ros2_bag2_home_rgbd_orbframe/gsplat/scene.ply`
- ORB voc: `third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt`
- 빌드 yaml(맵 생성에 쓴 것): `configs/ros2_bag2_home_rgbd_orbslam3.yaml`

---

### Task 1: localization-mode settings yaml

**Files:**
- Create: `configs/ros2_bag2_home_rgbd_localization.yaml`

**Interfaces:**
- Produces: ORB-SLAM3가 읽을 localization 설정 — 빌드 yaml과 동일 intrinsics, 단 `System.SaveAtlasToFile` 대신 `System.LoadAtlasFromFile`.

- [ ] **Step 1: 빌드 yaml 복제 후 Atlas 키 교체**

`configs/ros2_bag2_home_rgbd_orbslam3.yaml`을 복사해 `configs/ros2_bag2_home_rgbd_localization.yaml`로 만들고, 마지막 `System.SaveAtlasToFile: "ros2_bag2_home_rgbd"` 줄을 제거하고 아래로 교체:

```yaml
# 저장된 맵 로드(고정), 새로 저장 안 함 — localization 전용
System.LoadAtlasFromFile: "outputs/ros2_bag2_home_rgbd/slam/ros2_bag2_home_rgbd"
```

(주의: ORB-SLAM3는 확장자 `.osa`를 자동 부착하므로 경로에 `.osa`를 쓰지 않는다.)

- [ ] **Step 2: 키 검증**

Run: `grep -E "LoadAtlasFromFile|SaveAtlasToFile|Camera1.fx" configs/ros2_bag2_home_rgbd_localization.yaml`
Expected: `LoadAtlasFromFile` 1줄 존재, `SaveAtlasToFile` 없음, `Camera1.fx: 642.284` 유지.

- [ ] **Step 3: Commit**

```bash
git add configs/ros2_bag2_home_rgbd_localization.yaml
git commit -m "feat(reloc): localization-mode settings yaml (LoadAtlasFromFile)"
```

---

> **참고:** 원래 Task 2(C++ ORB localization 러너)·Task 3(ORB 실행)은 실제로 빌드·실행했으나 ORB-SLAM3 localization mode가 cold per-frame reloc에 부적합(atlas 서브맵 로드 시 639점만 활성 + RECENTLY_LOST 잠김 → 0% reloc)으로 **폐기**. Stage 1을 **COLMAP image_registrator 기반 per-frame localization**으로 교체(아래). 좌표가 orbframe(=가우시안 프레임)에 바로 떨어지고, 사용자가 고른 "프레임별 cold reloc" 의도에 부합. 폐기된 C++ 산출물(`rgbd_localization.cc`, CMake 타깃)은 남겨두되 미사용.

### Task 2: COLMAP query localization 파이프라인

**Files:**
- Create: `scripts/localize_query_colmap.sh`

**Interfaces:**
- Consumes: orbframe reference 모델(`data/processed/ros2_bag2_home_rgbd_orbframe/colmap/sparse/0/images.txt` = orbframe 좌표 known poses) + home rgb. `scripts/make_orb_seed_from_db.py` 재사용.
- Produces: registered 모델 `outputs/ros2_bag2_home_rgbd/reloc_pnp/registered/`(TXT) — reference 224 KF + 등록된 query 프레임 포즈, 전부 orbframe 좌표. query 등록 성공 수.

**핵심 아이디어:** orbframe reference의 known poses를 **고정**해 point_triangulator로 descriptor 있는 모델을 orbframe 좌표에 만들고, query 프레임을 `image_registrator`로 그 모델에 등록 → query 포즈가 orbframe(=가우시안) 좌표로 나온다.

- [ ] **Step 1: localize_query_colmap.sh 작성**

```bash
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

# reference 224 KF 이름
grep -v '^#' "$REF/images.txt" | awk 'NF>=10 && NR%2==1{print $10}' | sort > "$OUT/ref_names.txt"
# query = reference 아닌 home rgb 프레임에서 STRIDE 간격으로 NQ개(시간순)
ls "$RGB" | sort -t. -k1 -n | grep -vxFf "$OUT/ref_names.txt" | awk "NR%$STRIDE==1" | head -n "$NQ" > "$OUT/query_names.txt"
cat "$OUT/ref_names.txt" "$OUT/query_names.txt" | sort -u > "$OUT/all_names.txt"
echo "[loc] ref $(wc -l < "$OUT/ref_names.txt")  query $(wc -l < "$OUT/query_names.txt")"

# 1) 특징 추출(ref+query) + exhaustive 매칭
"$COLMAP" feature_extractor --database_path "$DB" --image_path "$RGB" \
  --image_list_path "$OUT/all_names.txt" \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$INTR" --FeatureExtraction.use_gpu 0
"$COLMAP" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0

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
```

- [ ] **Step 2: 실행 (백그라운드)**

```bash
chmod +x scripts/localize_query_colmap.sh
nohup bash scripts/localize_query_colmap.sh 80 4 > outputs/ros2_bag2_home_rgbd/reloc_pnp/run.log 2>&1 < /dev/null &
```

- [ ] **Step 3: 완료 대기 + query 등록 성공률**

Run:
```bash
until grep -qE "registered 모델|ERROR|error" outputs/ros2_bag2_home_rgbd/reloc_pnp/run.log; do sleep 10; done
REG=$(grep -vc '^#' outputs/ros2_bag2_home_rgbd/reloc_pnp/registered/images.txt)
echo "등록 이미지(ref+query) 절반=$((REG/2)); query 목표 $(wc -l < outputs/ros2_bag2_home_rgbd/reloc_pnp/query_names.txt)"
```
Expected: registered/images.txt에 ref(224)+등록된 query. query가 0개 등록이면 매칭 부족 → STRIDE 줄이거나 query를 reference 시간대 근처로. 진행 전 query 등록 >0 확인.

- [ ] **Step 4: Commit**

```bash
git add scripts/localize_query_colmap.sh
git commit -m "feat(reloc): COLMAP per-frame query localization against orbframe model"
```

---

### Task 3: registered 모델 → TUM reloc pose 파일

**Files:**
- Create: `scripts/export_reloc_poses.py`
- Test: `scripts/test_export_reloc_poses.py`

**Interfaces:**
- Consumes: `outputs/ros2_bag2_home_rgbd/reloc_pnp/registered/images.txt`(orbframe 좌표), `query_names.txt`.
- Produces: `outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt` — query 프레임만, 각 줄 `ts tx ty tz qx qy qz qw state`(state 2=등록됨). Stage 2(Task 4·5)가 이 포맷·경로를 그대로 소비.
- 함수 `colmap_image_to_twc_tum(qw,qx,qy,qz,tx,ty,tz) -> (cx,cy,cz, qx,qy,qz,qw)`: COLMAP Tcw(qw qx qy qz tx ty tz)를 Twc TUM(카메라중심 + R_wc 쿼터니언 x,y,z,w)으로 변환.

- [ ] **Step 1: 실패 테스트 작성**

```python
# scripts/test_export_reloc_poses.py
import numpy as np
from export_reloc_poses import colmap_image_to_twc_tum

def test_identity_tcw_gives_origin_twc():
    # Tcw=identity → 카메라 중심 원점, 회전 identity
    cx,cy,cz, qx,qy,qz,qw = colmap_image_to_twc_tum(1,0,0,0, 0,0,0)
    assert np.allclose([cx,cy,cz],[0,0,0], atol=1e-6)
    assert np.allclose([qx,qy,qz,qw],[0,0,0,1], atol=1e-6)

def test_translation_tcw_center_is_negative_R_t():
    # Tcw t=(0,0,5), R=I → 카메라 중심 C=-R^T t=(0,0,-5)
    cx,cy,cz,*_ = colmap_image_to_twc_tum(1,0,0,0, 0,0,5)
    assert np.allclose([cx,cy,cz],[0,0,-5], atol=1e-6)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd scripts && python -m pytest test_export_reloc_poses.py -v`
Expected: FAIL (모듈/함수 없음).

- [ ] **Step 3: export_reloc_poses.py 구현**

```python
#!/usr/bin/env python3
"""export_reloc_poses.py — COLMAP registered 모델(orbframe 좌표)에서 query 프레임 포즈를 TUM Twc로 추출.
출력: outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt (ts tx ty tz qx qy qz qw state)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pipeline.backproject import read_colmap_images

def colmap_image_to_twc_tum(qw, qx, qy, qz, tx, ty, tz):
    R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()      # Tcw 회전
    C = -R_cw.T @ np.array([tx, ty, tz])                    # 카메라 중심(world)
    q = Rot.from_matrix(R_cw.T).as_quat()                   # R_wc → (x,y,z,w)
    return (*C, *q)

def main():
    reg = ROOT/"outputs/ros2_bag2_home_rgbd/reloc_pnp/registered/images.txt"
    qn = ROOT/"outputs/ros2_bag2_home_rgbd/reloc_pnp/query_names.txt"
    out = ROOT/"outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt"; out.parent.mkdir(parents=True, exist_ok=True)
    imgs = read_colmap_images(reg)                          # {name: (qw,qx,qy,qz,tx,ty,tz)}
    queries = [l.strip() for l in open(qn) if l.strip()]
    n = 0
    with open(out, "w") as f:
        for name in sorted(queries, key=lambda s: float(s.rsplit('.',1)[0])):
            ts = name.rsplit('.',1)[0]
            if name in imgs:                                # 등록 성공
                cx,cy,cz,qx,qy,qz,qw = colmap_image_to_twc_tum(*imgs[name])
                f.write(f"{ts} {cx:.7f} {cy:.7f} {cz:.7f} {qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f} 2\n"); n+=1
            else:                                           # 등록 실패 = LOST
                f.write(f"{ts} 0 0 0 0 0 0 1 3\n")
    print(f"[export] query {len(queries)}, 등록 {n} ({100*n/max(len(queries),1):.0f}%) → {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 단위테스트 통과 + 실행**

Run: `cd scripts && python -m pytest test_export_reloc_poses.py -v` → 2 PASS.
그다음: `conda run -n xrsplat python scripts/export_reloc_poses.py` → 등록률 출력, `CameraTrajectory_reloc.txt` 생성.
Expected: 등록률 >0%, 파일에 query 줄 존재.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_reloc_poses.py scripts/test_export_reloc_poses.py
git commit -m "feat(reloc): export COLMAP-registered query poses to TUM reloc file"
```

---

### Task 4: Stage 2 — render_reloc_replay.py (pose→viewmat 단위테스트 포함)

**Files:**
- Create: `scripts/render_reloc_replay.py`
- Test: `scripts/test_reloc_replay.py`

**Interfaces:**
- Consumes: reloc pose 파일(Task 3), home rgb, orbframe scene.ply, intrinsics.
- Produces: 함수 `twc_line_to_viewmat(tx,ty,tz,qx,qy,qz,qw) -> np.ndarray(4,4)` (Tcw); CLI가 `demo.mp4` 생성.

- [ ] **Step 1: 실패 테스트 작성 (pose→viewmat 규약)**

```python
# scripts/test_reloc_replay.py
import numpy as np
from render_reloc_replay import twc_line_to_viewmat

def test_identity_pose_gives_identity_viewmat():
    # Twc=identity(카메라 원점·정렬) → Tcw=identity
    vm = twc_line_to_viewmat(0,0,0, 0,0,0,1)
    assert np.allclose(vm, np.eye(4), atol=1e-6)

def test_translation_only_inverts():
    # 카메라가 world (1,2,3)에 있고 회전 없음 → Tcw t = -C
    vm = twc_line_to_viewmat(1,2,3, 0,0,0,1)
    assert np.allclose(vm[:3,3], [-1,-2,-3], atol=1e-6)
    assert np.allclose(vm[:3,:3], np.eye(3), atol=1e-6)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd scripts && python -m pytest test_reloc_replay.py -v`
Expected: FAIL (`ModuleNotFoundError: render_reloc_replay` 또는 함수 없음).

- [ ] **Step 3: render_reloc_replay.py 구현**

```python
#!/usr/bin/env python3
"""render_reloc_replay.py — reloc pose로 orbframe 가우시안 렌더 + 실제 rgb와 side-by-side 영상.
usage: render_reloc_replay.py --reloc <CameraTrajectory_reloc.txt> --scene ros2_bag2_home_rgbd_orbframe
       --rgb-scene ros2_bag2_home_rgbd [--start 0 --end -1 --stride 2 --width 640 --fps 18]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, cv2, imageio.v2 as imageio
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gsplat_io import load_ply, render
from pipeline.backproject import read_colmap_cameras
DEV = "cuda"; ROOT = Path(__file__).resolve().parents[1]

def twc_line_to_viewmat(tx, ty, tz, qx, qy, qz, qw):
    R_wc = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    R_cw = R_wc.T
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R_cw; vm[:3, 3] = -R_cw @ np.array([tx, ty, tz])
    return vm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reloc", required=True, type=Path)
    ap.add_argument("--scene", default="ros2_bag2_home_rgbd_orbframe")
    ap.add_argument("--rgb-scene", default="ros2_bag2_home_rgbd")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--out", type=Path, default=ROOT/"docs/assets/reloc_replay_home.mp4")
    a = ap.parse_args()
    proc = ROOT/"data/processed"/a.scene; rgbdir = ROOT/"data/processed"/a.rgb_scene/"rgb"
    W,H,fx,fy,cx,cy = read_colmap_cameras(proc/"colmap/sparse/0/cameras.txt")
    K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1.]], device=DEV)
    g = load_ply(ROOT/"outputs"/a.scene/"gsplat/scene.ply", DEV)
    lines = [l.split() for l in open(a.reloc) if len(l.split())>=9]
    end = len(lines) if a.end<0 else a.end
    lines = lines[a.start:end:a.stride]
    frames, ok = [], 0
    for s in lines:
        ts = s[0]; vals = list(map(float, s[1:8])); state = int(float(s[8]))
        rgb = cv2.imread(str(rgbdir/f"{ts}.png"))            # 실제 프레임
        if rgb is None: continue
        rgb = rgb[:, :, ::-1]
        if state == 2:                                       # reloc/track OK → 렌더
            vm = torch.tensor(twc_line_to_viewmat(*vals), device=DEV)
            with torch.no_grad():
                rnd = (render(g, vm, K, W, H).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            ok += 1
        else:                                                # LOST → 회색 + 텍스트
            rnd = np.full_like(np.asarray(rgb), 40)
            cv2.putText(rnd, "RELOC LOST", (rnd.shape[1]//4, rnd.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,80,80), 2)
        combo = np.hstack([np.asarray(rgb), np.full((H,4,3),255,np.uint8), rnd]).astype(np.uint8)
        cv2.putText(combo, "REAL", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(combo, "GAUSSIAN(reloc)", (W+14,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        if a.width != combo.shape[1]:
            h2 = round(combo.shape[0]*a.width/combo.shape[1])
            combo = cv2.resize(combo, (a.width, h2))
        frames.append(combo)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps)
    print(f"[replay] {len(frames)}프레임, reloc OK {ok} ({100*ok/max(len(frames),1):.0f}%) → {a.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 단위테스트 통과 확인**

Run: `cd scripts && python -m pytest test_reloc_replay.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_reloc_replay.py scripts/test_reloc_replay.py
git commit -m "feat(reloc): render_reloc_replay.py — reloc pose render + side-by-side video"
```

---

### Task 5: 좌표 일관성 검증 (reloc pose vs orbframe known pose ATE)

**Files:**
- Create: `scripts/check_reloc_frame.py`

**Interfaces:**
- Consumes: reloc pose 파일, orbframe `colmap/sparse/0/images.txt`.
- Produces: reloc pose와 같은 프레임의 orbframe 알려진 pose 간 ATE(cm). 작으면 reloc이 가우시안 맵과 같은 프레임에 떨어진 증거.

- [ ] **Step 1: 검증 스크립트 작성**

```python
#!/usr/bin/env python3
"""check_reloc_frame.py — reloc Twc vs orbframe known Twc ATE(같은 프레임). 좌표 일관성 증명."""
import sys; from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pipeline.backproject import read_colmap_images, colmap_world_RT

def reloc_centers(p):
    out = {}
    for l in open(p):
        s = l.split()
        if len(s) >= 9 and int(float(s[8])) == 2:
            out[s[0]] = np.array(list(map(float, s[1:4])))   # Twc translation = 카메라 중심
    return out

def main():
    reloc = reloc_centers(ROOT/"outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt")
    imgs = read_colmap_images(ROOT/"data/processed/ros2_bag2_home_rgbd_orbframe/colmap/sparse/0/images.txt")
    known = {n.rsplit('.',1)[0]: np.array(colmap_world_RT(*imgs[n])[1]) for n in imgs}
    common = sorted(set(reloc) & set(known))
    if not common:
        print("공통 프레임 0 — 타임스탬프 포맷 불일치 확인"); return
    d = np.array([np.linalg.norm(reloc[k]-known[k]) for k in common])*100
    print(f"공통 {len(common)}프레임  reloc vs known ATE: median {np.median(d):.1f}cm  p90 {np.percentile(d,90):.1f}cm")
    print("→ 작으면(~cm) reloc이 가우시안 맵과 같은 ORB 프레임에 정합 = decoupled 런타임 성립")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 + 일관성 확인**

Run: `conda run -n xrsplat python scripts/check_reloc_frame.py`
Expected: 공통 프레임 다수, ATE median 작음(맵·자산 동일 ORB 프레임이므로 cm급 기대). 큰 값(>50cm)이면 reloc 맵과 orbframe snap 기준이 어긋난 것 → 원인 조사.

- [ ] **Step 3: Commit**

```bash
git add scripts/check_reloc_frame.py
git commit -m "feat(reloc): coordinate-consistency check (reloc vs known pose ATE)"
```

---

### Task 6: end-to-end 데모 영상 생성 + 보고

**Files:**
- Create: `docs/assets/reloc_replay_home.mp4` (산출물)

**Interfaces:**
- Consumes: Task 3 reloc 로그, Task 4 스크립트.

- [ ] **Step 1: 데모 영상 생성 (GPU env, 백그라운드)**

```bash
nohup bash -c '
  source /home/ingon/miniconda3/etc/profile.d/conda.sh; conda activate xrsplat
  export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="8.9" CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
  exec python -u scripts/render_reloc_replay.py \
    --reloc outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt \
    --stride 2 --width 720 --fps 18
' > outputs/ros2_bag2_home_rgbd/reloc/replay.log 2>&1 < /dev/null &
```

- [ ] **Step 2: 완료 확인 + 대표 프레임 육안 검토**

Run: `until grep -q "replay\]" outputs/ros2_bag2_home_rgbd/reloc/replay.log; do sleep 10; done; cat outputs/ros2_bag2_home_rgbd/reloc/replay.log; ls -la docs/assets/reloc_replay_home.mp4`
Expected: 영상 생성, reloc OK % 출력. 대표 프레임에서 가우시안 렌더가 실제 프레임과 정렬되는지 육안 확인(side-by-side).

- [ ] **Step 3: Commit (스크립트/로그 정리, mp4는 용량 따라 gitignore 판단)**

```bash
# mp4가 크면 gitignore, 작으면 docs/assets에 커밋
git add -A docs/assets/.gitignore 2>/dev/null || true
git commit -m "docs(reloc): home runtime reloc-render replay demo" --allow-empty
```

---

## Self-Review 체크

- **Spec coverage:** Stage1(Task1-3) / Stage2(Task4) / 좌표일관성 검증(Task5) / 데모+성공률(Task3,6) / 에러 RELOC LOST(Task4) — 전부 태스크 있음.
- **Type consistency:** `twc_line_to_viewmat` 시그니처 Task4 정의·테스트 일치. reloc 파일 포맷(9열, state at [8]) Task2 생성 ↔ Task4·5 소비 일치. viewmat=Tcw 규약 Global Constraints와 일치.
- **리스크:** reloc 성공률이 낮으면 Task3 Step2에서 멈추고 원인(맵 로드) 조사. `ActivateLocalizationMode`/`LoadAtlasFromFile` 동작은 Task2-3 실행으로 실증.
