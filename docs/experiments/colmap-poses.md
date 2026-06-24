# 실험: COLMAP 포즈로 gsplat 학습 (experiment/colmap-poses)

> ⚠ 이건 **검증된 메인 경로가 아니라 실험/증거**다. SPEC의 ORB-SLAM3 테제(ORB 포즈 + 좌표계 공유 →
> XR 런타임 relocalization 맵)를 **fix한 게 아니라 우회**한다. main(SPEC)은 건드리지 않는다.

## 왜 (진단)
xr-splat 자체 캡처(room1/room2) .ply가 흐렸다. 통제 실험으로 원인을 분리:
- **트레이너 무죄**: 우리 트레이너 = RGB-only = 원본 gsplat simple_trainer, ORB 포즈면 다 흐림.
- **캡처도 (생각보다) 무죄**: 같은 room2인데 포즈만 COLMAP으로 바꾸면 선명.
- **범인 = ORB-SLAM3 포즈의 광학적 부정확**(~cm). COLMAP SfM 재투영오차 0.69px vs ORB는 트래킹+노이즈 depth로 ~3cm 불일치.
- **pose-opt 회복 실패**(lr 1e-5 무효 / 1e-4 무효 / 1e-3 발산). 광학적 pose-opt는 cm급 오차 교정 불가.

## 결과 (2-샘플, 재현됨)
| 장면 | ORB 포즈 (median PSNR) | COLMAP 포즈 (우리 파이프라인) | 개선 |
|---|---|---|---|
| room1 | 21.6 | **27.0** (mean 27.4, max 34) | +5.4dB |
| room2 | 20.5 | **25.8** (mean 24.9) | +5.3dB |

공통: 잘 덮인 영역(키보드·데스크) 선명, 가장자리·모니터 뷰는 여전히 흐림(캡처 커버리지 잔여).

## 레시피 (재현 절차)
1. **COLMAP SfM** (`scripts/run_colmap_sfm.sh`, 이 브랜치에서 exhaustive→sequential + bin→txt 수정).
   - 동적(사람) 프레임은 SfM에서 제외. **정적 세트는 시간 불연속이라 sequential 대신 exhaustive matcher** 필요(room1: 132/133 등록, 0.88px).
2. **bin→txt** (`colmap model_converter --output_type TXT`).
3. **scene 구성** — `setup_sfm_baseline.py`의 metric 스케일정렬은 **노이즈 depth 장면에서 spread WIDE**(room1 0.39 / room2 0.25)라 부적합 →
   대신 **COLMAP 네이티브 포즈 + 네이티브 sparse 점** 사용(`d455_<scene>_sfm/colmap/sparse/0/`에 images.txt·points3D.txt 복사).
4. **점 outlier 필터(필수)** — COLMAP sparse 점의 원거리 outlier가 우리 트레이너의 `init_gaussians` `scene_scale`
   (점 max-from-mean)을 부풀려 **means LR 폭주 → 발산**. 점을 **카메라 reach 이내로 필터**해 scene_scale를 카메라 스케일에 맞춤.
   (외부 simple_trainer는 scene 정규화로 자동 회피 — 우리 트레이너의 latent 약점.)
5. **학습**: `train_gsplat.py --scene d455_<scene>_sfm --depth-lambda 0 --refine-stop 7000 --iters 30000` (RGB-only).
   depth OFF 이유: COLMAP 네이티브는 임의 스케일 → metric depth 감독과 충돌. RGB-only가 robust(외부 24dB 레시피와 동일).

## 한계 / 미해결
- **임의 스케일**(metric 아님), **런타임 relocalization 맵 없음**, COLMAP 좌표계 ≠ ORB 맵 좌표계.
- 가장자리·모니터 블러는 **캡처 한계**(작은 공간 범위·발광 반사면·모션블러) — 포즈로 못 고침.
- **열린 문제**: ORB 런타임 맵 ↔ COLMAP 렌더 품질 화해 = **Sim3 1회 정렬**(둘 다 같은 키프레임 포즈 보유). XR 단계서 결정.

## 한 줄
**렌더 품질 = 포즈 품질이 지배. ORB 포즈가 병목, COLMAP이면 +5dB.** 단 이건 "방법(우회)"이고, 진짜 제품(ORB맵+COLMAP품질)은 미해결.
