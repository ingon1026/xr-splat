# 설계 — home 런타임 reloc→render 리플레이 데모

작성: 2026-06-24 · 상태: 승인됨(브레인스토밍) → 구현 플랜 대기

## Context (왜)

xr-splat decoupled 아키텍처의 핵심 약속은 **"SLAM pose로 가우시안 맵을 런타임 렌더"** 이다. 지금까지 M1·M2로 *오프라인 자산 생성*(프레임 유효 home 가우시안 맵, ORB 프레임 정합 scale 1.0/rot 0°)은 끝냈으나, **런타임 절반(고정 맵에 relocalize → 그 pose로 렌더)은 한 번도 돌리지 않았다.** 이 데모가 그 절반을 증명해 "통합 모델 버리고 분리한 게 옳았다"를 end-to-end로 보인다.

증명 형태(브레인스토밍 결정): **오프라인 리플레이 영상**, **프레임별 relocalization**(저장된 home `.osa` 맵에 cold-reloc), 출력은 **실제 프레임 | 가우시안 렌더 나란히**.

## 핵심 통찰 (좌표 일관성)

reloc pose는 `outputs/ros2_bag2_home_rgbd/slam/ros2_bag2_home_rgbd.osa`(ORB 맵 프레임)에서 나온다. home 가우시안 자산 `ros2_bag2_home_rgbd_orbframe`은 바로 그 ORB 런(`ros2_bag2_home_rgbd`)에 snap 정합돼 있다(scale 1.0/rot 0° 검증 완료). **따라서 reloc pose를 렌더러에 변환 없이 그대로 꽂는다** — 지금까지의 프레임 정합 작업이 여기서 보상받는다.

## 아키텍처 (2단계, pose 파일 결합)

```
home rgb 프레임 + home.osa
   │
 [Stage 1: ORB-SLAM3 localization mode]   맵 고정, 새 KF 없음, 프레임별 reloc
   │  → CameraTrajectory_reloc.txt (TUM Twc) + reloc 성공/실패 플래그
   ▼
 [Stage 2: render_reloc_replay.py]
   │  각 pose로 orbframe scene.ply 렌더 + 실제 rgb와 side-by-side + 상태 오버레이
   ▼
 demo.mp4 / demo.gif
```

두 단계는 **on-disk pose 파일**로만 통신 → 독립 실행·교체 가능.

## 컴포넌트

### Stage 1 — localization-mode RGB-D 러너 (작은 C++)
- `third_party/ORB_SLAM3/Examples/RGB-D/rgbd_tum.cc`를 복제한 `rgbd_localization.cc`:
  - settings yaml에 `System.LoadAtlasFromFile: <home 맵 경로>` (맵 로드).
  - 추적 루프 진입 전 `SLAM.ActivateLocalizationMode()` (새 KF·맵 변경 차단 = 고정 맵 reloc).
  - 프레임별 `TrackRGBD` 결과 pose와 추적 상태(OK/LOST)를 기록.
- 기존 ORB-SLAM3 빌드가 동작하므로 CMake에 타깃 1개 추가 후 재빌드.
- WSLg 규칙 준수: **뷰어 OFF(headless)**, 성공 판정은 종료코드 아닌 출력 파일.
- 산출: `outputs/<scene>/reloc/CameraTrajectory_reloc.txt` (frame_ts tx ty tz qx qy qz qw status).

### Stage 2 — `scripts/render_reloc_replay.py` (신규)
- 입력: reloc pose 파일, home rgb 디렉토리, `outputs/ros2_bag2_home_rgbd_orbframe/gsplat/scene.ply`, intrinsics.
- 각 프레임: pose가 유효(reloc OK)면 `pipeline.gsplat_io.render(g, view_matrix, K, W, H)`로 렌더; 실패면 렌더 자리에 "RELOC LOST".
- 실제 rgb | 가우시안 렌더 가로 결합 + 프레임 라벨(ts, reloc 상태) 오버레이 → `imageio`로 mp4/gif 인코딩.
- 재사용: `gsplat_io.load_ply/render`, `backproject.read_colmap_cameras`, `render_teaser.py`의 인코딩 패턴.

## 데이터 흐름

`home rgb + home.osa` → Stage1 → `CameraTrajectory_reloc.txt` → Stage2(+ rgb + orbframe scene.ply) → `demo.mp4`.

## 입력 범위

home 연속 walk 구간 **~10–15초**, **2–3프레임당 1장** 다운샘플(부드러운 영상, 렌더 비용↓). 전체 1206 프레임은 데모에 불필요. 구간·스트라이드는 스크립트 인자로 조정.

## 에러 처리

- **프레임별 reloc 실패는 정상적 결과** — 실패 프레임은 렌더 없이 "RELOC LOST" 표시.
- 데모 끝에 **reloc 성공률**(붙은 프레임 %) 리포트. 낮아도 숨기지 않음.
- Stage 1이 맵 로드 실패/0 reloc면 Stage 2 진입 전 중단하고 원인 보고.

## 검증 (end-to-end)

1. **reloc 성공률**: 몇 % 프레임이 고정 맵에 relocalize 됐나.
2. **좌표 일관성 (수치)**: reloc pose vs 같은 프레임의 orbframe 알려진 pose **ATE** — 작으면 reloc이 가우시안 맵과 같은 프레임에 떨어진 증거(decoupled 런타임 성립).
3. **시각**: side-by-side 영상에서 가우시안 렌더가 실제 프레임과 정렬.
4. 배경 실행(긴 작업 nohup), pkill 자살 패턴 금지.

## 리스크 / 비고

- Stage 1 C++ 러너 + 재빌드 필요(리스크 낮음 — 기존 빌드 동작). `ActivateLocalizationMode`/`LoadAtlasFromFile` API는 설치본으로 확인 후 확정.
- reloc 성공률이 낮을 수 있음(같은 시퀀스라 유리하나 cold-reloc은 시점·텍스처 의존). 낮으면 그 자체가 정직한 한계 결과.
- 렌더 품질은 기존 home 자산 그대로(soft) — 데모는 **좌표/메커니즘 증명**이 목적, crispness는 별도 트랙.
- 관련: [[xrsplat-sfmsnap-frame-integrity]], SPEC §Phase 6.
