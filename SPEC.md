# XR-Splat: Decoupled SLAM + Gaussian Splatting 파이프라인 구현 명세

> 이 문서는 Claude Code에게 전달되는 구현 명세서다. 위에서부터 Phase 순서대로 구현하고,
> 각 Phase의 Acceptance Criteria를 통과한 뒤 다음 Phase로 진행할 것.

---

## 0. 프로젝트 개요

### 목표
XR에서 사람이 들어가도 어색하지 않은 실사급 공간 자산(3D Gaussian Splatting 모델, .ply)을 구축한다.

### 핵심 아키텍처 결정 (변경 금지)
SLAM과 Gaussian Splatting을 **분리(decoupled)** 한다.

- **로컬라이제이션**: ORB-SLAM3 (RGB-D 모드) — 키프레임 포즈 추정 + 맵 저장(런타임 relocalization용)
- **실사화**: gsplat (Nerfstudio 계열 라이브러리) — SLAM이 준 포즈를 **고정 입력**으로 받아 학습
- 두 맵은 동일한 SLAM 포즈로 만들어지므로 **좌표계를 자동 공유**한다. 별도 정합 불필요.

배경: SplaTAM / GS-SLAM / Photo-SLAM / LoopSplat 등 coupled 방식을 실험한 결과
렌더링 품질·트래킹 정확도·속도 모두 미달 → 역할 분리로 전환. (연구노트 별도 문서 참조)

### 데이터 흐름
```
D455 캡처(.bag) → TUM RGB-D 포맷 추출 → ORB-SLAM3 실행
  → KeyFrameTrajectory.txt (TUM 포맷, Twc)
  → 변환기: TUM → COLMAP 포맷 (Tcw 역변환 + 쿼터니언 순서 변경)
  → depth 역투영 포인트클라우드 (가우시안 초기화용)
  → gsplat 학습 (포즈 고정 + depth loss)
  → 후처리 (pruning / SH 축소 / 압축)
  → 최종 산출물: scene.ply (+ ORB-SLAM3 Atlas 맵 = 런타임 relocalization용)
```

---

## 1. 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble), WSL2 (kernel 6.6.123.2-microsoft-standard-WSL2+) on Windows 11 |
| GPU | NVIDIA GeForce RTX 4070 Ti · 12GB VRAM (12282 MiB) · 드라이버 591.86 · CUDA(driver) 13.1 · WSL CUDA 동작 확인 ✓ (nvidia-smi/nvcc) |
| CUDA toolkit | 시스템 nvcc 13.1 (현행 torch/gsplat 휠과 비호환 가능) → `xrsplat` conda env에 CUDA 12.x toolkit 격리(13.1 드라이버 하위호환). `torch.cuda.is_available()`는 Phase 0에서 확인 |
| 카메라 | Intel RealSense D455 (자체 캡처 .bag 미보유 → M2 단계 전 촬영) |
| Python | 3.10+ (conda, 환경 이름: `xrsplat`) |
| 디스크 / RAM | 442 GB 여유 / 1007 GB (54% 사용) · RAM 31 GB (+ swap 8 GB) |

### 1.1 WSL2 + RealSense 전략 (중요)

WSL2에서 RealSense 라이브 스트리밍은 알려진 난관이다. usbipd로 USB를 연결해도
`Cannot access /sys/class/video4linux` 류의 오류로 카메라가 인식되지 않는 사례가 흔하다
(WSL2 커널에 UVC 비디오 모듈이 없기 때문).

**Plan A (기본 채택): Windows 캡처 + WSL2 오프라인 처리**
1. Windows에서 Intel RealSense Viewer로 녹화 → `.bag` 파일 생성
2. `.bag`을 WSL2 파일시스템으로 복사 (`/mnt/c/...` 경유보다 WSL 내부 디스크로 복사하는 것이 I/O 빠름)
3. WSL2의 librealsense(또는 pyrealsense2 pip 패키지)로 `.bag`을 오프라인 파싱

이 파이프라인은 어차피 오프라인 처리이므로 라이브 스트리밍이 필요 없다. Plan A로 충분하다.
pyrealsense2는 `pip install pyrealsense2`로 설치하고, `rs.config.enable_device_from_file()`로
.bag 재생이 카메라 없이 동작하는지 먼저 검증할 것.

**Plan B (Plan A 실패 시에만): usbipd-win + RSUSB 백엔드 소스 빌드**
- Windows: `usbipd bind/attach --wsl` 로 D455 연결, WSL에서 `lsusb`로 8086:0b5c 확인
- librealsense를 커널 패치 없이 빌드: `cmake .. -DFORCE_RSUSB_BACKEND=ON`
- 권한 문제 발생 시 udev rules 설치(99-realsense-libusb.rules). 그래도 sudo가 필요한 사례 있음.
- Plan B는 시간 소모가 크므로 30분 이상 막히면 Plan A로 복귀할 것.

### 1.2 D455 권장 캡처 설정 (문서화하여 capture 가이드에 포함)
- 해상도: color 1280x720 @ 30fps, depth 848x480 @ 30fps (D455 depth 최적 해상도는 848x480)
- **auto-exposure / auto-white-balance OFF** (수동 고정) — 프레임 간 색 일관성 확보
- depth와 color의 해상도가 다르므로 **align(depth→color)** 처리 필수 (rs.align)
- depth scale: D455는 기본 1mm 단위 (depth_scale=0.001). 코드에 하드코딩하지 말고 SDK에서 읽을 것
- 촬영 요령: 천천히 이동, 같은 영역 다각도, 시작 지점으로 복귀(루프 클로저 유도)

---

## 2. 저장소 구조 (이대로 스캐폴딩할 것)

```
xr-splat/
├── README.md                  # 섹션 8 참조
├── LICENSE                    # MIT (자체 코드에 한함, 섹션 8.4 라이선스 주의 참조)
├── .gitignore                 # 섹션 8.2
├── .gitattributes             # LFS 사용 시
├── environment.yml            # conda 환경 정의 (재현성)
├── docs/
│   ├── research-note.md       # 연구 방향 정리 문서 (기존 작성본 이관)
│   ├── capture-guide.md       # D455 촬영 가이드
│   └── assets/                # README용 이미지/GIF (각 5MB 이하)
├── configs/
│   ├── d455_orbslam3.yaml     # ORB-SLAM3 설정 (스크립트가 자동 생성)
│   └── gsplat_default.yaml    # 학습 하이퍼파라미터
├── scripts/                   # 파이프라인 각 단계 (모두 CLI, argparse)
│   ├── 01_extract_bag.py      # .bag → TUM RGB-D 폴더 구조
│   ├── 02_run_orbslam3.sh     # ORB-SLAM3 실행 래퍼
│   ├── 03_tum_to_colmap.py    # TUM 궤적 → COLMAP 포맷 (핵심 변환기)
│   ├── 04_make_pointcloud.py  # depth 역투영 초기 포인트클라우드
│   ├── 05_validate_poses.py   # 변환 검증 ("벽 한 장" 테스트)
│   ├── 06_train_gsplat.sh     # gsplat 학습 래퍼
│   ├── 07_evaluate.py         # PSNR/SSIM/LPIPS 평가
│   └── 08_postprocess.py      # pruning / SH 축소 / 압축
├── pipeline/                  # 스크립트들이 공유하는 모듈 (포맷 변환, 기하 유틸)
├── third_party/               # git submodule (직접 복사 금지)
│   └── ORB_SLAM3/             # submodule
├── data/                      # gitignore 대상. 구조만 .gitkeep으로 유지
│   ├── raw/                   #   .bag 원본
│   ├── processed/             #   TUM 포맷, COLMAP 포맷
│   └── README.md              #   데이터 입수 방법 안내 (이 파일은 커밋)
└── outputs/                   # gitignore 대상 (.ply, 체크포인트, 렌더링)
```

원칙: **모든 단계는 독립 실행 가능한 CLI 스크립트**로 만들고, 입출력은 디스크의 표준 포맷
(TUM, COLMAP)으로만 주고받는다. 한 단계를 다른 구현으로 교체해도 나머지가 영향받지 않게 한다.

---

## 3. Phase별 구현 지시

### Phase 0 — 스캐폴딩 & 환경
1. 위 저장소 구조 생성, git init, 첫 커밋
2. environment.yml 작성 (python 3.10, numpy, opencv-python, open3d, pyrealsense2, torch+CUDA, gsplat, nerfstudio)
3. WSL2에서 CUDA 동작 확인 (`torch.cuda.is_available()`)
4. ORB-SLAM3를 submodule로 추가하고 빌드:
   - 의존성: Pangolin, OpenCV 4.x, Eigen3
   - 알려진 빌드 이슈: 최신 Ubuntu에서 C++ 표준 불일치 컴파일 에러가 흔함.
     CMakeLists에서 `-std=c++14` 관련 수정이 필요할 수 있음. 빌드 에러는 에러 메시지 기준으로
     커뮤니티에 알려진 패치를 적용하되, **수정 사항은 patch 파일로 `third_party/patches/`에 보관**
     (submodule 내부를 직접 커밋할 수 없으므로)
   - WSL2 GUI: WSLg가 있으면 Pangolin 뷰어가 뜬다. 안 뜨면 뷰어 비활성(headless) 옵션으로 실행
5. **Acceptance**: ORB-SLAM3가 TUM RGB-D 공개 시퀀스(fr1/desk 등) 예제로 정상 실행되어
   KeyFrameTrajectory.txt를 출력한다.

### Phase 1 — .bag → TUM RGB-D 추출 (`01_extract_bag.py`)
입력: `.bag` / 출력: TUM RGB-D 디렉토리 구조
```
<scene>/
├── rgb/<timestamp>.png        # color, 8bit 3ch
├── depth/<timestamp>.png      # 16bit 1ch, mm 단위 (TUM 관례: 5000 스케일이 아니라
│                              #   ORB-SLAM3 yaml의 DepthMapFactor와 일치시킬 것. 1000 권장)
├── rgb.txt / depth.txt        # "timestamp filename" 목록
├── associations.txt           # rgb-depth 타임스탬프 매칭 (TUM associate.py 로직 내장)
└── intrinsics.json            # fx fy cx cy, width height, depth_scale, distortion(k1 k2 p1 p2 k3)
```
구현 요건:
- rs.align으로 depth를 color 프레임에 정렬한 후 저장 (intrinsics는 color 스트림 것을 사용)
- 왜곡계수(distortion: k1 k2 p1 p2 k3)도 intrinsics.json에 기록 — 원본 이미지 렌즈왜곡 보정(ORB-SLAM3 트래킹 정확도). D455 bag=SDK coeffs, TUM=공식계수, Replica=0
- 프레임 드랍 대비: timestamp 기반 매칭, 매칭 실패 프레임은 버리고 카운트 로깅
- intrinsics.json은 이후 모든 단계의 단일 진실 공급원(single source of truth)
- **Acceptance**: 추출된 rgb/depth 한 쌍을 open3d로 역투영했을 때 장면 형태가 보인다.

### Phase 2 — ORB-SLAM3 실행 (`02_run_orbslam3.sh` + yaml 자동 생성)
- intrinsics.json에서 ORB-SLAM3 yaml(`d455_orbslam3.yaml`)을 자동 생성하는 헬퍼 포함
  (Camera.fx 등 + DepthMapFactor를 Phase 1의 depth 스케일과 반드시 일치)
  - **distortion 모델 분기**: ORB-SLAM3 `Camera1.k1..k3/p1/p2`는 forward Brown-Conrady(radtan) 계수를 기대한다.
    RealSense color 스트림은 `inverse_brown_conrady`(역모델)로 보고하므로 그 계수를 radtan에 직매핑하면 부정확하다.
    D455 color는 공장 정류로 계수가 ~0이라, **역모델이면 계수 0 처리**(forward radtan/brown_conrady는 그대로 사용).
- RGB-D 모드로 실행, 산출물 2개를 outputs/<scene>/slam/에 보관:
  1. `KeyFrameTrajectory.txt` (TUM 포맷: timestamp tx ty tz qx qy qz qw — **Twc, 즉 camera-to-world**)
  2. Atlas 맵 파일 (`System.SaveAtlasToFile` 설정 활성화) — 런타임 relocalization 자산
- CameraTrajectory.txt(전체 프레임)가 아니라 **KeyFrameTrajectory.txt(키프레임)** 를 후속 단계에 사용
  (루프 클로저/전역 BA 반영이 가장 정확한 포즈이며, gsplat 학습엔 키프레임이면 충분)
- **Acceptance**: 자체 캡처 시퀀스에서 트래킹 로스 없이 완주하고 궤적이 출력된다.
  트래킹 로스 발생 시 ORBextractor.nFeatures 증가(1250→2000) 등 파라미터 조정 가이드를 문서화.

### Phase 3 — TUM → COLMAP 변환 (`03_tum_to_colmap.py`) ★ 최다 실수 구간
출력: COLMAP text 모델 (`cameras.txt`, `images.txt`, `points3D.txt`) + 이미지 폴더 심링크/복사

**좌표 변환 명세 (정확히 이대로):**
- TUM 한 줄 = (t, tx ty tz, qx qy qz qw) = **Twc** (camera→world)
- COLMAP images.txt 한 줄 = (IMAGE_ID, qw qx qy qz, tx ty tz, CAMERA_ID, NAME) = **Tcw** (world→camera)
- 변환: `R_cw = R_wc^T`, `t_cw = -R_wc^T @ t_wc`, 쿼터니언 출력 순서는 **(qw, qx, qy, qz)**
- 회전/쿼터니언 연산은 scipy.spatial.transform.Rotation 사용 (손 구현 금지)
- cameras.txt: PINHOLE 모델, intrinsics.json 값 사용
- 키프레임 timestamp ↔ 이미지 파일명 매칭은 허용 오차(예: 10ms) 내 최근접 매칭

**함정 체크리스트 (코드 주석과 테스트에 반영):**
1. Twc→Tcw 역변환 누락 → 학습이 "되는 것처럼 보이며 흐릿하게" 실패
2. 쿼터니언 순서: TUM은 (qx qy qz qw), COLMAP은 (qw qx qy qz)
3. COLMAP images.txt는 이미지당 **2줄**(2번째 줄은 2D points, 빈 줄 허용) — 포맷 위반 시 파서 침묵 실패
4. scipy Rotation의 quat 순서는 (x,y,z,w) — 입출력 시 명시적으로 재배열

**단위 테스트 필수**: 합성 포즈(알려진 R, t)로 왕복 변환(TUM→COLMAP→TUM) 항등성 검증.

### Phase 4 — 초기 포인트클라우드 (`04_make_pointcloud.py`) & 검증 (`05_validate_poses.py`)
- N개(기본 30) 키프레임의 depth를 Tcw 포즈로 월드 좌표에 역투영, 색은 rgb에서 샘플
  - **max-depth 클램프(기본 6m)**: D455 신뢰범위(~0.6~6m) 밖 원거리 depth는 스테레오 노이즈/uint16 포화(65.5m)라
    역투영 시 떠도는 점이 되므로 무효화(0)한다. backproject가 depth>0만 사용. (실측 room1: max 65.5m outlier 관측)
- voxel downsample (기본 2cm) 후 points3D.txt(또는 .ply)로 저장 → gsplat 초기화 입력
- `05_validate_poses.py`: **공통 표면이 겹치는 시점 페어**를 골라 각각 역투영해 정합 검사.
  **판정 기준: 같은 벽/바닥이 한 겹이면 PASS, 두 겹 유령이면 FAIL.**
  - 페어 선정·겹침 마스크는 **신뢰 Twc**(KeyFrameTrajectory)로 한다 — 검증 대상인 Tcw로 잡으면 진짜
    ghost가 게이트 밖으로 빠져 false-PASS 됨. **정합 측정은 COLMAP Tcw 역투영**으로(gsplat이 쓰는 포즈).
  - baseline은 **moderate(기본 0.2~0.6m)** — 너무 작으면 약한 게이트, 너무 크면 SLAM 드리프트가 임계를 잠식.
  - 수치 판정: 겹침 영역 최근접거리의 **percentile(p50) < 3cm**(평균 아님 — 비겹침 점이 평균 오염).
    **중앙 크롭(기본 0.6)이 1차 포즈 판정**(PINHOLE+렌즈왜곡이 full-frame을 지배), full-frame은 왜곡 진단.
  - 페어를 못 찾으면 **cannot-validate**(절대 default-PASS 금지).
  - 파라미터(baseline/percentile 임계/crop)는 **TUM 책상 스케일 기본값** — Replica/D455 룸스케일은 재검증 필요.
    - **룸스케일 knobs 원칙**: **baseline·overlap-min은 궤적 스케일에 맞춰 조정 가능**(근접/소이동 스캔은 baseline 하한↓·overlap↓해
      co-visible 페어를 확보). 단 **판정 임계 `pass-cm`(=3cm)은 고정** — 게이트 강도 자체는 절대 약화하지 않는다(페어 *선정* 범위만 조정).
      실측 D455 room1(총 이동 1.09m, 중앙 depth 0.64m): `--baseline 0.1 0.28 --overlap-min 0.15`에서 PASS(center p50 0.9~2.3cm).
      seam(트래킹 로스→맵 병합) 구간은 **seam을 가로지르는 페어**로 별도 검사할 것(강체 offset은 baseline과 무관하게 p50에 드러남).
  - 한계: NN거리는 in-plane 평행이동에 둔감(gross 역변환은 평면 밖이라 잡힘). off-by-one 배선버그는
    association(kf_ts==rgb_ts, ‖C−t_wc‖≈0)으로 별도 배제. negative control(포즈 섭동→FAIL)로 게이트 작동 확인.
- **Acceptance**: 05 스크립트 PASS. 이 검증 전에는 절대 Phase 5로 진행하지 말 것.

### Phase 5 — gsplat 학습 (`06_train_gsplat.sh`)
- 기본 경로: nerfstudio `splatfacto` (COLMAP 데이터 직접 로드) 또는 gsplat `simple_trainer` 중
  설치/호환성이 깔끔한 쪽을 택일. 선택 근거를 README에 기록
- **필수 설정**:
  - 카메라 포즈 최적화 OFF (포즈 고정) — splatfacto의 경우 camera-opt 비활성 플래그 확인
  - depth supervision ON — Phase 1의 depth를 데이터셋에 포함시켜 depth loss 활성화
  - iteration 30k 기본 (오프라인이므로 품질 우선)
- **동적/제외 프레임**: `--exclude-list`로 사람 등 동적 구간 KF를 04 init·06 학습 양쪽에서 제외(유령 방지).
- 산출물: `outputs/<scene>/gsplat/scene.ply` + 학습 로그 + 렌더링 샘플 이미지
- **Acceptance**: 학습 완료 후 테스트 뷰 렌더링이 입력 사진과 시각적으로 일치.
  - **발산 감시(필수)**: train_log.jsonl의 **step 8000~12000**에서 `depth_l1`·`n_gauss` 안정 확인
    (M1 발산: refine-stop 없으면 step~9000에 depth_l1 급등 + gaussians 폭주). **제외 비율이 높을수록**
    (무감독 영역↑) 더 위험. **발산 시 회피(파라미터 조정) 금지 — 근본 원인 조사를 연다.**

### Phase 6 — 평가 (`07_evaluate.py`) & 후처리 (`08_postprocess.py`)
- 평가: 학습에서 제외한 hold-out 키프레임(매 8번째)에 대해 PSNR/SSIM/LPIPS 산출, JSON 저장
- **베이스라인 비교 실험 (M1 핵심 근거)**: 동일 이미지로 (a) COLMAP SfM 포즈 vs (b) ORB-SLAM3 포즈 학습 → 품질 차이.
  공정 비교를 위해 **포즈 소스만 다르고 나머지는 전부 동일**해야 한다:
  - **스케일 처리**: COLMAP SfM는 스케일 임의(monocular). metric depth loss와 충돌하므로 **COLMAP 재구성을 metric으로
    스케일 정렬**한 뒤 비교한다 — 정렬 계수 `s = median(metric_depth / COLMAP_sparse_depth)` (가시 sparse, **유효 depth 픽셀만**).
    **비율 분포가 tight한지 확인·스프레드 리포트** — bimodal/wide면 스케일 드리프트·매칭 불량(blind median 금지). s를
    포즈 translation·포인트에 적용. 그 후 **양쪽 동일 프로토콜**(metric 포즈 → 04 dense-depth init → 06 depth ON).
    (양쪽 depth-off는 스케일 회피엔 간단하나 depth-supervised 파이프라인 자체를 검증 못 하므로 비채택.)
  - **hold-out 분할**: 매 8번째 키프레임을 평가용 제외, **양쪽 동일 인덱스**. hold-out은 **양쪽 모두 포즈가 있는
    공통(교집합) 프레임**에서만 — 한쪽만 등록된 프레임으로 평가하면 비교가 오염됨. **register-all-then-split**: COLMAP
    SfM·SLAM은 hold-out 프레임의 포즈도 산출해야 한다(렌더해 채점) — 학습에서만 제외, 포즈 추정에선 제외하지 않음.
  - **등록 실패 프레임**: COLMAP SfM 미등록·ORB 미추적 프레임은 비교 평가에서 **양쪽 모두 제외**(교집합 평가),
    제외 개수와 등록률(ORB vs COLMAP)을 리포트 — 등록률 자체가 견고성 근거.
  - **동일 프로토콜**: 동일 hold-out·동일 학습 config/iter/해상도·동일 평가(렌더 후 PSNR/SSIM/LPIPS).
  - **학습 프로토콜 (M1 실측값)**: 양쪽 **15k iter + `--refine-stop 7000`(densification 조기 중단) + means grad clip(max_norm 10)** + 공통 **hold-out 16뷰**.
    ⚠ **hold-out 학습은 step ~9000에서 발산**한다(제외된 뷰 영역의 가우시안이 무감독 → drift → densification 폭주 7~10M). 위 안전장치 필수.
    **발산 원인은 완전 규명이 아니라 안전장치로 회피**한 상태 — **ORB 경로만 발산**(COLMAP은 동일 설정에서 안정)했으므로, **M2(자체 캡처)에서 재발 시 근본원인 재조사**할 것.
  - **M1 판정**: 절대 PSNR이 아니라 **차이 |PSNR_ORB − PSNR_COLMAP| ≤ 0.5dB**, **per-view 분포로 리포트**(mean-of-means
    금지 — 한 경로가 특정 뷰에서만 망가지는 걸 평균이 숨김). PSNR-Δ는 간접 proxy라 단독으론 약한 baseline과의 일치도
    통과시킴 → **ATE-vs-GT 필수**: TUM `groundtruth.txt`로 양쪽 포즈의 절대 정확도(ATE)를 측정·리포트해 **COLMAP이
    신뢰할 gold standard임을 먼저 입증**. M1 PASS는 PSNR-Δ와 ATE 둘 다로 해석한다.
- 공개 데이터셋 선행 검증: 자체 캡처 전에 TUM RGB-D 또는 Replica 시퀀스로 전체 파이프라인
  1회 통과 (GT 궤적과 ATE 비교 가능)
- 후처리: opacity 기반 pruning → SH degree 3→1 축소 옵션 → 압축 export.
  각 단계 전후의 (가우시안 수, 파일 크기, PSNR) 표 자동 출력
- **좌표계 무결성 게이트 (sfmsnap 자산 채택 전 필수)**: sfmsnap(SfM 포즈를 ORB metric 프레임에 스냅) 자산은
  **PSNR만으로 고르면 안 된다** — PSNR은 씬 자기 포즈로 렌더하므로 좌표계가 틀어져도 높게 나온다. 채택 전
  **sfmsnap 포즈 vs ORB base 포즈의 Sim3 적합이 scale≈1.0 & rot≈0°**(이미 동일 프레임)인지 확인한다.
  틀어지면(예: 공통 KF 부족으로 정합 실패) 런타임 ORB relocalization이 맵을 못 찾아 **decoupled가 무용**이 된다.
  실측(2026-06-23): room2 sfmsnap = scale 1.0000/rot 0.00° PASS(자산 채택), home sfmsnap = PSNR 더 높지만
  rot 75.78° FAIL(보류). 통과해도 남는 per-pose 보정량(room2=5.8cm)은 빌드/런타임 오프셋으로 리포트.
- **깨진 프레임 수정 (`snap_scene_to_orb.py`)**: FAIL한 sfmsnap도 재캡처 없이 고친다 — ORB와 타임스탬프 매칭되는
  공통 프레임으로 Sim3를 적합해 src 전체 포즈+포인트를 ORB 프레임에 강체 재정렬 후 재학습. 실측(2026-06-24 home):
  76° 깨짐 → 수정 후 scale 1.0/rot 0°, **holdout PSNR 23.82**(깨진 sfmsnap 24.34 대비 −0.52dB) = 프레임 유효 M2 자산.
- **05 게이트 scene-scale 보정 + 음성대조**: 대형 D455 scene(home median depth 2~3m)은 depth 노이즈 바닥(3~7cm)이
  룸 책상 기본 `pass-cm 3.0` 위라 부적합 → `--baseline 0.15 0.4 --pass-cm 4.0`로 보정(PASS/FAIL이 depth와 상관함을
  증명). **보정이 게이트를 약화 안 시켰음을 음성대조로 증명: images.txt(Tcw)를 회전3°+TZ10cm 섭동 → FAIL 해야 함**
  (05는 in-plane translation·KeyFrameTrajectory 섭동엔 둔감하므로 그 둘로 음성대조하면 거짓 PASS — 틀린 음성대조).
- **densification 폭주 주의**: snap/변형 씬은 default `refine-stop 15000`에서 가우시안 폭주 가능(home orbframe
  step6700 5.7M, GPU OOM직전). 정상 plateau 기준(sfmsnap=2.0M 고정)과 비교 판단 — 폭주 시 `--refine-stop`을 입증
  카운트(~2M)에 맞춰 조기 정지(M1도 hold-out 발산에 동일). 대형 공간은 GPU 메모리가 밀도를 제한해 small scene보다 soft.

---

## 4. 전역 코딩 규칙
- 모든 스크립트: argparse + `--help` 완비, 진행 로그는 logging 모듈, 실패 시 비정상 종료코드
- 경로 하드코딩 금지. 장면 단위 디렉토리 규약: `data/processed/<scene>/`, `outputs/<scene>/`
- 좌표계를 다루는 모든 함수는 docstring에 입력/출력 컨벤션(Twc/Tcw, quat 순서) 명시
- 수치 검증이 가능한 곳(Phase 3)은 pytest 단위 테스트 작성
- 커밋: conventional commits (`feat:`, `fix:`, `docs:`, `test:`), Phase 단위 브랜치 후 main 머지

---

## 5. GitHub 공개 전략

### 5.1 핵심 원칙: "저장소 = 재현 방법, 결과물 = 별도 배포"
이 프로젝트의 산출물(.ply, .bag)은 수백 MB~수 GB의 바이너리다. git은 텍스트 버전 관리 도구라
바이너리를 커밋하면 모든 버전이 히스토리에 누적되어 저장소가 영구히 비대해진다.
**git에는 "그 결과물을 만드는 방법"(코드, 설정, 문서, 스크립트)만 올린다.**
잘 만든 연구 저장소의 기준은 "클론 → 환경 구성 → 스크립트 실행만으로 동일한 .ply가
재현되는가"이지, 결과 파일이 들어있는가가 아니다.

### 5.2 .gitignore (필수 항목)
```
data/raw/
data/processed/
outputs/
*.bag
*.ply
*.osa
*.ckpt
*.pth
__pycache__/
.venv/
wandb/
```
단, `data/README.md`(데이터 입수 방법 안내)와 각 디렉토리의 `.gitkeep`은 커밋한다.

### 5.3 큰 파일이 꼭 필요할 때의 선택지 (우선순위순)
1. **GitHub Releases**: 대표 결과물(데모용 scene.ply 1~2개)을 버전 태그와 함께 첨부.
   대용량 바이너리 배포의 공식 권장 경로이며 저장소 용량을 차지하지 않는다.
2. **외부 호스팅**: 데이터셋·다수의 결과물은 Hugging Face Datasets(연구 표준, 무료) 또는
   클라우드 스토리지에 올리고 README에서 링크 + 다운로드 스크립트 제공
3. **Git LFS는 최소한으로**: LFS는 큰 파일을 포인터로 대체해 별도 서버에 저장하는 방식인데,
   무료 플랜은 저장소 10GiB·월 대역폭 10GiB 한도가 있어 .bag/.ply 같은 파일을 넣으면 금방
   소진된다. README용 데모 GIF·티저 이미지처럼 작고 필수적인 바이너리에만 제한적으로 사용
4. 절대 금지: data/, outputs/ 통째 커밋, 100MB 초과 파일 직접 커밋(GitHub가 push 거부)

### 5.4 서드파티와 라이선스 (중요)
- ORB-SLAM3는 **GPLv3**: 소스를 저장소에 복사해 넣지 말고 **git submodule**로 참조.
  우리 코드는 ORB-SLAM3를 별도 프로세스로 실행(파일 입출력 연동)하므로 자체 코드는
  MIT 등으로 유지 가능. README에 서드파티 라이선스 고지 섹션 작성
- gsplat은 Apache 2.0, 원조 Inria 3DGS 코드는 비상업 연구용 제한 — 원조 코드를 쓰게 되면 고지
- 수정이 필요한 submodule은 patch 파일(`third_party/patches/`) + 적용 스크립트로 관리

### 5.5 README.md 구성 (이 순서로 작성)
1. 한 줄 소개 + **티저**: 최종 렌더링 GIF 또는 before/after 이미지 (시각 결과물 프로젝트의 README는 첫 화면이 전부다)
2. 파이프라인 다이어그램 (mermaid 또는 docs/assets 이미지)
3. Why decoupled?: coupled 방식 실험 결과와 전환 근거 3줄 요약 (상세는 docs/research-note.md 링크)
4. Installation: WSL2 전제 명시, environment.yml, ORB-SLAM3 빌드 (+ 흔한 빌드 에러 트러블슈팅)
5. Usage: 캡처 → 01~08 스크립트를 순서대로 실행하는 Quick Start (복붙 가능한 명령어 블록)
6. Results: 정량 표 (COLMAP 포즈 vs ORB 포즈 PSNR/SSIM/LPIPS, ATE) + 데모 .ply 다운로드 링크(Releases)
7. Repository structure, Third-party licenses, Acknowledgements

### 5.6 공개 전 체크리스트
- [ ] fresh clone 후 README만 보고 환경 구성이 되는가 (가능하면 새 WSL 인스턴스에서 점검)
- [ ] `git count-objects -vH` 로 저장소 크기 50MB 이하 확인
- [ ] 히스토리에 대용량 파일이 들어간 적 없는지 확인 (들어갔다면 공개 전에 history rewrite)
- [ ] data/README.md에 자체 캡처 데이터 없이 공개 데이터셋(TUM/Replica)으로 재현하는 경로 안내
- [ ] 개인정보: 자체 캡처 .bag/이미지에 얼굴·주소 등이 있으면 공개 자산에서 제외

---

## 6. 마일스톤 요약 (진행 순서)

| # | 작업 | 완료 판정 |
|---|---|---|
| M0 | 스캐폴딩 + ORB-SLAM3 빌드 | TUM 공개 시퀀스 예제 구동 |
| M1 | 공개 데이터셋 E2E | TUM/Replica 1개 시퀀스가 01→07 전체 통과, **공통 hold-out·동일 프로토콜**에서 COLMAP 대비 **PSNR 차 \|ORB−COLMAP\| ≤ 0.5dB**(절대값 아님) **+ 양쪽 ATE-vs-GT 리포트**(COLMAP이 신뢰할 baseline인지) |
| M2 | D455 자체 캡처 E2E | 자체 촬영 방 1개의 scene.ply 생성, 05 검증 PASS. **1차(room1): 파이프라인 검증 통과**(05 PASS·발산 없이 scene.ply 생성, 동적 사람 KF는 `--exclude-list`로 04/06 제외) — **단 품질은 궤적 협소(총 이동 1.09m)로 한계(정적뷰 PSNR ~21.6, 흐림). 재캡처 1회 예정**(궤적 3m+, 제자리회전 금지). |
| M3 | 후처리 + 결과 정리 | 경량화 표 + README Results 섹션 완성 |
| M4 | 공개 | 5.6 체크리스트 전부 통과 후 public 전환 |

M1을 M2보다 먼저 한다. 공개 데이터셋은 GT가 있어 "내 촬영 문제"와 "파이프라인 문제"를
분리해주기 때문이다. M1이 통과되기 전에 자체 캡처 디버깅에 시간을 쓰지 말 것.
