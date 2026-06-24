# xr-splat — 슬라이드 구성 (기승전결)

> 통합형 Gaussian-SLAM에서 **decoupled(SLAM + Gaussian 분리)** 로의 전환과 end-to-end 실증
> 작성 2026-06-24 · 실측 수치 기반

---

## Slide 1 — 표지
**XR 실사급 공간 자산을 위한 SLAM × Gaussian Splatting**
부제: 통합형의 한계와 **decoupled 구조**로의 전환 — 설계부터 런타임 실증까지

---

## Slide 2 — 목표 (起)
**"XR에서 사람이 공간에 들어가도 어색하지 않은 실사급 공간 자산"**
- 보이는 화면 = 실사 같은 3D 공간
- 동시에 = 사용자 위치를 안정적으로 추정(localization)
- 핵심 질문: **이걸 한 모델로? 아니면 나눠서?**

---

## Slide 3 — 첫 시도: 통합형 Gaussian-SLAM (起)
하나의 Gaussian 표현으로 tracking · mapping · rendering을 **실시간 동시** 수행
- 실험: **SplaTAM · GS-SLAM · Photo-SLAM · LoopSplat** 4종 직접 빌드·실행
- 기대: "한 시스템으로 위치추정 + 실사 렌더 다 해결"

---

## Slide 4 — 통합형의 한계 (承)
4종 실험 결과 — **세 가지가 동시에 미달**
- 🔴 렌더 품질: 실사급 미달
- 🔴 트래킹: 성숙한 전용 SLAM보다 약함
- 🔴 속도: 무거움 (실시간 XR 부담)
- → **"장점 통합"이 아니라 "단점 통합"** 위험. 한 표현에 모든 역할을 욱여넣은 대가.

---

## Slide 5 — 전환 결정: decoupled (承 → 轉)
**역할을 쪼갠다**
- **Localization → ORB-SLAM3** (성숙한 포즈·루프클로저·relocalization)
- **Rendering → gsplat** (오프라인 무제약 실사 최적화)
- 핵심 트릭: **Gaussian 맵을 SLAM 포즈 위에서 학습** → 두 맵이 **동일 좌표계** → 런타임 정합 0단계

---

## Slide 6 — decoupled 파이프라인 (轉)
```
[오프라인] RGB-D 캡처 → ORB-SLAM3 포즈 → COLMAP → gsplat 학습(포즈 고정+depth) → 경량 자산
[런타임]  새 프레임 → 저장된 맵에 relocalize → 그 pose로 Gaussian 렌더
```
- 8단계 독립 CLI, 표준 포맷(TUM/COLMAP)으로 통신 → 단계 교체 자유

---

## Slide 7 — M1: 원리 검증 (轉, 공개 데이터)
**"SLAM 포즈로도 실사급 GS가 나오나?"** — TUM fr1/desk, hold-out 16뷰
- ORB 포즈 **23.85** vs COLMAP 포즈 **23.98 PSNR** → 차이 **0.13dB**
- ATE 둘 다 ~2cm
- ✅ **decoupled 원리 성립** — 좋은 입력이면 SLAM 포즈로 COLMAP급 품질

---

## Slide 8 — M2: 실제 D455 캡처 (轉)
자체 캡처 3개 — 캡처 기하가 품질을 좌우
| 캡처 | 궤적 | 성격 |
|---|---|---|
| room1 | 0.43m | 제자리회전, narrow |
| room2 | 0.65m | 좁은 영역 |
| **home** | **28.78m** | 공간 횡단, baseline 충분 |
- 교훈: **걸으면서 시차(parallax) 확보**가 실사 품질의 전제

---

## Slide 9 — 난관 1: 포즈 품질이 병목 (轉)
room1에서 발견 — 같은 캡처인데 **포즈를 바꾸니 +5dB**
- ORB 포즈 vs full SfM 포즈: per-pose **10cm** 차이
- SfM 포즈로 학습 시 **+5dB** 선명
- → 실제 캡처에선 **ORB 포즈 정확도가 실사 품질의 병목**

---

## Slide 10 — 난관 2: 프레임 정합 함정 (轉, 핵심 발견)
**"PSNR 높다고 좋은 자산이 아니다"**
- sfmsnap(SfM 품질 + ORB 좌표계 스냅) 전략 도입
- 함정: home은 PSNR 24.34로 최고였지만 **ORB 프레임에서 76° 틀어짐** → 런타임 위치추정 불가 = 무용
- 해결: `snap_scene_to_orb.py`로 ORB 프레임에 강체 재정렬 → scale 1.0/rot 0°
- 결과: **home 23.82 PSNR + 프레임 유효** (−0.52dB로 좌표계 회복) — 재캡처 0

---

## Slide 11 — 런타임 reloc → render 실증 (轉 → 結)
**decoupled의 존재 이유를 end-to-end로 증명**
- 저장된 맵에 새 프레임 **relocalize**(COLMAP PnP) → 그 pose로 Gaussian 렌더
- reloc 성공 **10/10 (100%)**
- 좌표 앵커 오차 **0.000cm** (포즈가 가우시안 맵 프레임에 정확히 정합)
- 시각: **실제 화면 = 가우시안 렌더** (TV·식물·소파·창 동일 위치)
- 〔이미지: `docs/assets/reloc_demo_still.png` / 영상: `reloc_replay_home.mp4`〕

---

## Slide 12 — 결과 종합 (結)
| 단계 | 결과 |
|---|---|
| 통합형 4종 | 한계 확인(렌더·트래킹·속도) |
| decoupled 원리(M1) | ✅ Δ0.13dB |
| 실사 자산(M2 home) | ✅ 23.82, 프레임 유효 |
| 런타임 reloc→render | ✅ 10/10, 앵커 0cm |
→ **"통합 모델보다 분리가 옳았다"를 처음부터 끝까지 데이터로 증명**

---

## Slide 13 — 남은 한계 & 다음 (結)
- ⚠️ **시각 품질은 "알아보지만 soft"** — 실사급엔 한 칸 부족
  - 원인: 대형 공간(home 28m) + GPU 12GB가 가우시안 밀도 제한 (처리 아닌 **메모리 천장**)
- 다음 후보: scene 타일링(품질) / 잘 찍은 캡처 / XR 뷰어 통합
- 단, **연구 방향(분리가 맞다)은 결론남**

---

## Slide 14 — 결론 (結)
> 통합형 Gaussian-SLAM은 이론은 매력적이나 렌더·트래킹·속도를 동시에 만족 못 함.
> **SLAM(위치) + Gaussian(렌더)을 분리하고 동일 좌표계로 연동**하면,
> 안정적 localization 위에서 실사 공간을 렌더하는 구조가 **실제로 작동한다.**
- 설계 → 자산 → 런타임까지 **전 구간 실증 완료**
