# xr-splat — 최종 성능 지표 종합

decoupled SLAM(ORB-SLAM3) × Gaussian(gsplat) 파이프라인. 모든 수치는 실측·저장된 json 기준.

## 起 — 원리 검증 (M1, 공개데이터 TUM fr1/desk, mocap GT 있음)
| 지표 | 값 |
|---|---|
| ORB 포즈 vs COLMAP 포즈 PSNR 차이 | **Δ0.13 dB** |
| ATE vs mocap GT (ORB) | **1.9 cm** |
→ 좋은 입력이면 SLAM 포즈로 COLMAP급 품질. *유일한 mocap-GT 절대 정확도 증명.*

## 承 — 실사 자산 품질 (M2, 자체 D455 home)
| 자산 | 가우시안 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Default | 2.24M | 23.82 | 0.785 | 0.391 |
| **MCMC-2M (채택)** | 2.0M | **27.96** | **0.871** | **0.267** |
→ 학습 전략(MCMC)으로 같은 예산에서 배치 최적화. 병목은 raw-count 아닌 배치. 타일링 불필요.

## 轉 — 합쳐진 맵 성능
**포즈 허용오차** (render-vs-real, 기준 25.84 dB): 회전 1° → **18.91 dB(−7)**, 이동 1cm → 23.01 dB(−2.8). → localizer 예산 **<1°·~1cm**.

**수렴 basin** (photometric, 수렴=<1cm): 이동 5cm **100%**·20cm 75%·40cm 50% / 회전 3° **100%**·12° 75%·24° 38%.

**가우시안 맵 자가 위치추정** (photometric): 5cm/3° → **0.18cm/0.05°, 10/10 수렴**, render PSNR 15.55→25.91.

**70/30 동작 범위(envelope)**: *찍은 공간 안* = localize✓·render✓(28)·photometric✓ / *안 찍은 공간* = localize✓(SfM 0.73cm)·render✗(10)·photometric✗. → **율속 = 캡처 커버리지**, 위치추정 아님.

**feature-PnP 실시간 localizer** (non-KF, 맵에 없는 새 프레임):
| 지표 | 값 |
|---|---|
| 전역 reloc 성공 | **100%** (hint 없이) |
| 속도 | **26.7 FPS** (실시간 근처, 광도법 대비 8×) |
| render-vs-real (풀자산) | **28.3 PSNR** |

**풀 28.8m 맵 전역**: non-KF 40개 전역 localize **40/40**, 28 FPS, render-vs-real 22 PSNR (밀도↔범위 trade-off). → 큰 공간 전역 작동 증명.

## 結 — 시스템화 & 자동 판정
**XR-ready 자동 게이트** (build_report, 자동 Sim3 프레임무결성): home **XR_READY**(scale 0.999/rot 0.054°, reloc 100%), room2 RENDER_ONLY(프레임 유효, reloc 미측정).

**런타임 루프**: 40프레임 트래킹 40/40 OK, 상태기계(OK/LOST) 작동.

**실행 통합** (단일 CLI): `xrsplat build/report/view/localize/run`. 오프라인 8단계 → 1명령(게이트·resume 자동). **pytest 28 passed**, gsplat-free 불변식 유지.

## 산출물
- 3D 자산: `scene.ply`(2M, 풀) + `scene_lite.ply`(배포). home 2m 자산 + 풀 28.8m 자산.
- 결과 이미지: `docs/assets/report/{quality_mcmc_AB, pose_sensitivity_curve, convergence_basin, photometric_reloc_AB, localize_to_render, localization_full_map, home_gallery, home_vs_room2, merged_map_summary, ...}.png`

## 남은 한계 (정직)
- 자체 데이터 절대-cm GT 없음(mocap) → render-vs-real+궤적으로 증명. 절대 ATE는 TUM(M1=1.9cm)만.
- 실시간 ~27 FPS(30 근접), 오프라인 리플레이 — 실물 HMD/VIO 미연결.
- 품질↔범위 trade-off: 풀 28.8m은 밀도 퍼져 soft(22). 둘 다 높이려면 더 촘촘히 캡처/학습.
