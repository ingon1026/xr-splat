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

### Task 2: localization-mode RGB-D 러너 (C++)

**Files:**
- Create: `third_party/ORB_SLAM3/Examples/RGB-D/rgbd_localization.cc`
- Modify: `third_party/ORB_SLAM3/CMakeLists.txt` (새 실행 타깃 추가)

**Interfaces:**
- Consumes: Task 1 yaml, home.osa, associations.txt.
- Produces: 실행파일 `rgbd_localization`. 사용법 `./rgbd_localization <voc> <settings> <sequence_dir> <association> <out_reloc.txt>`. 출력 `out_reloc.txt` 각 줄 = `timestamp tx ty tz qx qy qz qw state` (state: 2=OK/tracked·reloc, 그 외=LOST). pose는 Twc(camera→world, world 좌표) TUM 규약.

- [ ] **Step 1: rgbd_localization.cc 작성**

`rgbd_tum.cc`를 기반으로, ① 생성자 직후 `SLAM.ActivateLocalizationMode()` 호출, ② 매 프레임 `TrackRGBD` 반환 pose와 `GetTrackingState()`를 out 파일에 직접 기록(SaveTrajectoryTUM은 LOST 프레임을 누락하므로 미사용). 핵심부:

```cpp
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include "System.h"
using namespace std;
// LoadImages: rgbd_tum.cc와 동일 (associations 파싱) — 그대로 복사
void LoadImages(const string&, vector<string>&, vector<string>&, vector<double>&);

int main(int argc, char **argv){
    if(argc != 6){ cerr << "Usage: ./rgbd_localization voc settings seq assoc out_reloc.txt\n"; return 1; }
    vector<string> vRGB, vD; vector<double> vT;
    LoadImages(string(argv[4]), vRGB, vD, vT);
    int n = vRGB.size();
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, false); // headless
    SLAM.ActivateLocalizationMode();                                          // 맵 고정, 새 KF 없음
    ofstream f(argv[5]); f << fixed;
    for(int ni=0; ni<n; ni++){
        cv::Mat imRGB = cv::imread(string(argv[3])+"/"+vRGB[ni], cv::IMREAD_UNCHANGED);
        cv::Mat imD   = cv::imread(string(argv[3])+"/"+vD[ni],   cv::IMREAD_UNCHANGED);
        if(imRGB.empty()) continue;
        double t = vT[ni];
        Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imD, t);
        int state = SLAM.GetTrackingState();                  // 2 = OK
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f tw = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        f << setprecision(6) << t << " " << setprecision(7)
          << tw.x()<<" "<<tw.y()<<" "<<tw.z()<<" "
          << q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" " << state << "\n";
    }
    f.close();
    SLAM.Shutdown();
    return 0;
}
```

(`LoadImages` 함수 본문은 `rgbd_tum.cc`의 것을 그대로 복사해 파일 하단에 둔다 — 엔지니어가 다른 태스크를 안 봐도 되게.)

- [ ] **Step 2: CMakeLists에 타깃 추가**

`third_party/ORB_SLAM3/CMakeLists.txt`에서 기존 `rgbd_tum` 타깃 정의를 찾아(`grep -n rgbd_tum CMakeLists.txt`) 그 바로 아래에 동일 패턴으로 추가:

```cmake
add_executable(rgbd_localization Examples/RGB-D/rgbd_localization.cc)
target_link_libraries(rgbd_localization ${PROJECT_NAME})
```

- [ ] **Step 3: 빌드 (백그라운드)**

```bash
cd third_party/ORB_SLAM3/build
nohup bash -c 'cmake --build . --target rgbd_localization -j4' > /tmp/build_reloc.log 2>&1 < /dev/null &
```

- [ ] **Step 4: 빌드 성공 확인**

Run: `until [ -x third_party/ORB_SLAM3/Examples/RGB-D/rgbd_localization ] || grep -qi error /tmp/build_reloc.log; do sleep 5; done; ls -l third_party/ORB_SLAM3/Examples/RGB-D/rgbd_localization`
Expected: 실행파일 존재. 에러 시 `/tmp/build_reloc.log` 확인.

- [ ] **Step 5: Commit**

```bash
git add third_party/ORB_SLAM3/Examples/RGB-D/rgbd_localization.cc third_party/ORB_SLAM3/CMakeLists.txt
git commit -m "feat(reloc): RGB-D localization-mode runner (per-frame pose+state)"
```

---

### Task 3: Stage 1 실행 — home reloc pose 생성

**Files:**
- Create: `outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt` (산출물, gitignore)

**Interfaces:**
- Consumes: Task 2 실행파일, Task 1 yaml, associations.txt.
- Produces: 프레임별 reloc pose 로그(Twc TUM + state). reloc 성공률 수치.

- [ ] **Step 1: localization 실행 (백그라운드, headless)**

```bash
mkdir -p outputs/ros2_bag2_home_rgbd/reloc
B=third_party/ORB_SLAM3
nohup bash -c "$B/Examples/RGB-D/rgbd_localization \
  $B/Vocabulary/ORBvoc.txt configs/ros2_bag2_home_rgbd_localization.yaml \
  data/processed/ros2_bag2_home_rgbd/rgb data/processed/ros2_bag2_home_rgbd/associations.txt \
  outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt" \
  > outputs/ros2_bag2_home_rgbd/reloc/run.log 2>&1 < /dev/null &
```

- [ ] **Step 2: 완료 대기 + reloc 성공률 확인**

Run:
```bash
until grep -qiE "shutdown|saving|error|terminate" outputs/ros2_bag2_home_rgbd/reloc/run.log; do sleep 10; done
awk '{c++; if($9==2) ok++} END{printf "frames %d, reloc/track OK %d (%.0f%%)\n", c, ok, 100*ok/c}' \
  outputs/ros2_bag2_home_rgbd/reloc/CameraTrajectory_reloc.txt
```
Expected: 프레임 다수에서 state==2(OK). 성공률 0%면 맵 로드 실패 → run.log에서 `LoadAtlasFromFile` 라인 확인(맵 경로/버전 불일치). 진행 전 반드시 >0%.

- [ ] **Step 3: Commit (코드 없음 — 산출물은 gitignore, 로그만 기록)**

산출물은 커밋하지 않는다. 성공률을 다음 태스크 검증의 입력으로 보고만 한다.

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
