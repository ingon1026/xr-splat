# xr-splat 운영 가이드 (셀프 인수인계)

> 이 파일을 레포의 docs/operating-guide.md 로 커밋해둘 것.
> 설계 배경과 진행 규칙은 SPEC.md / CLAUDE.md가 진실 공급원이고,
> 이 문서는 "사람(나)이 Claude Code를 어떻게 굴리는가"만 담는다.

---

## 0. 지금 즉시 할 일 (대화 끊기기 전 체크)

- [ ] 다운로드 확인: ① 연구방향 정리 md ② 구현명세 md(=SPEC.md 원본) ③ README.md
      — 셋 다 레포에 들어가면 원본 파일은 없어도 됨
- [ ] 연구방향 정리 md를 Claude Code에 드래그앤드랍:
      "이거 docs/research-note.md 로 넣고 커밋해줘"
- [ ] 이 파일을 드래그앤드랍: "이거 docs/operating-guide.md 로 넣고 커밋해줘"

## 1. 미처리 항목 (다음 세션에서 바로 던질 것)

```
하나만 확인하고 가자: 저번에 "advisor 지적 반영"이라고 보고했는데,
실제 advisor 모델 호출이 일어난 거야, 아니면 네 자체 검토였어?
실제 호출이면 advisor 설정이 어디 살아있는지 찾아서 보고해 —
우리 그거 끈 상태여야 해.

답 주고 나서 Phase 2 가자. headless 빌드 승인. 빌드는 백그라운드로.
```

- README 푸시가 아직이면: README 커밋/푸시 + 커밋 author 전수검사
  (Co-Authored-By 트레일러 금지, author=ingon1026) 먼저.
- 레포는 private 유지. public 전환은 M4 체크리스트(SPEC §5.6) 통과 후.

## 2. 평상시 운영 멘트 (이게 전부)

| 상황 | 멘트 |
|---|---|
| 기본 진행 | `계속해` 또는 `Phase N 가자` |
| 학습/빌드 중 상태 확인 | `백그라운드 작업 상태 어때` |
| Phase 끝나고 대화 길어짐 | `/clear` 후 → `SPEC.md, CLAUDE.md, docs/operating-guide.md 읽고 git log로 진행 상황 파악한 다음 이어서 해` |
| Phase 시작 | 플랜 모드(Shift+Tab) 켜고 `Phase N 계획부터 보여줘. 예상되는 에러와 대응도 미리 적어봐` |
| 완료 보고 받으면 | `Acceptance Criteria를 실제 실행 증거(로그/출력)로 보여주고, 통과면 커밋해` |

## 3. 보고 판정 요령 (내가 게이트키퍼인 지점)

- "통과했다"는 말만 있고 로그/수치 증거가 없으면 → 증거 요구. 양보 금지.
- SPEC에 없는 걸 추가/변경하겠다고 하면 → "SPEC.md에 먼저 반영하고 구현해".
- 아키텍처(decoupled, ORB-SLAM3, gsplat, 포즈 고정)를 다시 열려고 하면 → 거부. 확정 사항.
- 절대 양보 금지 게이트 2개:
  1. 05_validate_poses.py PASS 전에 gsplat 학습 시작 금지 (좌표 변환 검증)
  2. M1(공개 데이터셋 E2E, COLMAP 대비 PSNR 차 ≤ 0.5dB) 전에 자체 캡처 디버깅 금지
- Phase 3 완료 시: 서브에이전트 적대적 리뷰(SPEC 함정 체크리스트 4개) 후 커밋 — CLAUDE.md에 규칙 있음.

## 4. 진행 현황 스냅샷 (2026-06-12 기준)

- 완료: **Phase 0~6 전부, M1 PASS, M3 완료** (공개 데이터셋 fr1/desk E2E). ORB vs COLMAP **PSNR Δ=0.13dB(≤0.5)**, **ATE ORB 1.9cm / COLMAP 2.0cm** → decoupled 파이프라인 타당성 입증.
  - Phase 2 headless 127 KF+Atlas / Phase 3 단위+적대적리뷰 / Phase 4 05 게이트 PASS / Phase 5 gsplat(포즈고정+depth, 30k 23dB) / Phase 6 평가+후처리.
  - **M3**: README Results 표(median, ATE) + 각주(fr1/desk·hold-out 16뷰·15k+refine-stop 7000) 채움. 티저는 `render_teaser.py`로 30k 모델 fly-through GIF(키프레임 Slerp+Lerp 보간, 끝 floater 구간 트리밍, 1.95MB≤2MB, LFS 미사용 직접 커밋). 각주 링크는 outputs/ gitignore라 `07_evaluate.py` 참조로.
- 확정된 환경/구현 사실: Ubuntu 24.04/WSL2, RTX 4070 Ti 12GB, torch 2.1.2/cu121
  - gsplat 1.5.3 (prereq: cuda-cccl=12.1.55, setuptools<70). **gsplat hold-out 학습은 `--refine-stop 7000` + means grad clip(max_norm 10) 필수** — 없으면 step ~9000 발산(densification 폭주).
  - WSLg에서 ORB-SLAM3 뷰어 항상 크래시 → headless 고정, 판정은 출력 파일 기준 (CLAUDE.md §4)
  - RealSense 라이브 스트리밍 WSL2 비신뢰 → Windows RealSense Viewer로 .bag 녹화 (Plan A). `docs/capture-guide.md`에 녹화 설정·동선 정리.
  - **01 bag 모드는 Phase 1에서 librealsense 테스트 bag으로 실행 검증됨**(SDK brown_conrady coeffs 확인). Viewer Record와 추출기는 동일 SDK rosbag 스키마라 토픽/메타데이터 일치. **미검증 경로 = Viewer 녹화본 + 해상도 불일치 align**(테스트 bag은 동일 해상도였을 수 있음). 방어 보강: 모션(IMU) 프레임은 종료 대신 스킵, 미지원 color 포맷은 명시적 에러. align 타깃·저장 intrinsics 모두 color 스트림 기준(01_extract_bag.py:112,115).
  - COLMAP 베이스라인: `run_colmap_sfm.sh`(고정 TUM intrinsics, 127/127 등록) + `setup_sfm_baseline.py`(metric 스케일정렬 **s=0.234**, IQR/median 0.03)
  - 08 후처리: opacity prune→SH 3→1→경량, **421MB→105MB(-0.48dB)**
- **M2 1차(room1) — 파이프라인 검증 통과, 품질은 궤적 협소로 한계. 재캡처 1회 예정.**
  - E2E 완주: 추출(2175프레임, 1280×720, align 848→1280 ✓) → 02 ORB(트래킹 로스 1회 **→ 맵 병합 복구**, 303 KF) →
    03 → 05 PASS(룸스케일 baseline 0.1~0.28, seam 가로지르기 검사 ~1cm 용접) → 04/06 **사람 170 KF 제외**(`--exclude-list`, (c) 전략) →
    30k 발산 없이 완주(refine-stop 7000+grad clip; **N 722k 동결, M1 발산 재발 안 함** — 제외 56%에도).
  - 품질: 정적뷰 PSNR **full 21.6 / lite 21.4**(−0.22dB, 722k→338k, 179→35MB). **흐림** — 원인은 파이프라인 아닌 **캡처**(총 이동 1.09m·범위 0.285m 제자리 회전, D455 depth 노이즈, 모션블러).
  - **원인 확정 실험**: 정적 KF 시작/중간/끝 렌더 vs GT 비교 → **가장자리(start 18.2/end 19.1) ≫ 중앙(21.3) 블러 패턴**.
    잘 덮인 중앙만 그나마 재구성, 가장자리는 본 시점 부족 → **시차/커버리지 결핍 확정**(모니터 발광/반사면이 최악). 파이프라인 무결.
  - **room1 산출물 위치**(전부 outputs/=gitignore): `data/processed/d455_room1/`(추출 2175쌍+colmap),
    `outputs/d455_room1/slam/{KeyFrameTrajectory.txt(303KF), d455_room1.osa}`,
    `outputs/d455_room1/gsplat_exclude.txt`(사람 170KF), `outputs/d455_room1/gsplat/{scene.ply 722k/171MB, scene_lite.ply 338k/35MB(SuperSplat용)}`.
  - ⚠ **08 ref-view 함정**: 08은 첫 KF를 ref로 쓰는데 그게 **제외된 사람 프레임**이면 PSNR가 가짜(9.47)로 찍힌다. 실측은 정적뷰로 별도 측정.
- **조사 결과(2026-06-15) — 렌더 품질은 "포즈 품질"이 지배** (SPEC 변경 아님, 사실 기록):
  - 통제 실험으로 확정: 트레이너 무죄(우리=RGB-only=원본 gsplat 동일), 캡처도 (생각보다) 무죄 → **범인은 ORB-SLAM3 포즈의 ~cm 광학 부정확**.
    같은 데이터에 **포즈만 COLMAP SfM(0.69px)으로 교체 시 room1 21.6→27.0 / room2 20.5→25.8 (둘 다 +5dB)**. `pose-opt` 회복은 실패(lr 무효/발산).
    → 이전 "M2 흐림=캡처 한계" 결론은 **부분 정정**: 캡처도 영향이나 **더 큰 병목은 ORB 포즈**였음(잘 덮인 영역은 COLMAP 포즈서 선명, 가장자리·모니터만 캡처 잔여).
  - ⚠ 이건 **"해결"이 아니라 "우회"** — COLMAP은 임의 스케일·런타임 맵 없음, SPEC의 ORB 테제(좌표계 공유)와 모순. **main/SPEC 불변.**
  - 코드·레시피·상세는 **브랜치 `experiment/colmap-poses`**(`docs/experiments/colmap-poses.md`). 우리 트레이너 약점도 기록: COLMAP sparse outlier가 `scene_scale` 부풀려 means LR 발산 → 점 필터 필요.
  - **열린 문제(미해결)**: ORB 런타임 맵 ↔ COLMAP 렌더 품질 화해 = **Sim3 1회 정렬**(둘 다 같은 키프레임 포즈 보유). XR 런타임 단계서 결정.
- 다음: **M2 재캡처**(capture-guide 교훈 반영: 걸으며 궤적 3m+, 제자리회전 금지, 시작/끝 사람 프레임 인 금지). 도착 시 `01_extract_bag.py bag` → 02~08.
- 미해결: **ORB 경로만 hold-out 발산**(COLMAP은 동일설정에서 안정) — 안전장치로 회피했으나 근본원인 미규명, M2 재발 시 조사.
- GitHub: ingon1026/xr-splat (private). 푸시 전 히스토리 대용량 파일 검사 필수 (특히 ORBvoc).

## 5. 나중에 필요해질 지식 (대화에서 나온 결론 요약)

- 캡처 요령: 노출/화밸 수동 고정, 천천히, 다각도, 루프 닫기. D455 depth는 848x480이 최적.
- XR 배포: 연구 단계는 PCVR 테더링 + Unity(aras-p UnityGaussianSplatting VR 포크)로 검증
  → 이후 스탠드얼론(Quest 3, 예산 수십만 가우시안)은 pruning → SH 3→1 → 압축(.spz) → SuperSplat 정리.
- 런타임 정합: 가우시안 맵이 SLAM 포즈로 학습됐으므로 두 맵은 동일 좌표계.
  PCVR이면 ORB-SLAM3 Localization Mode(Atlas 로드)로 상시 트래킹,
  스탠드얼론이면 세션 시작 시 1회 relocalization 앵커 + 이후 디바이스 VIO.
- 결과물 배포: 레포에는 코드/문서만. 데모 .ply는 GitHub Releases, 데이터셋류는 Hugging Face.
  Git LFS는 README용 GIF 정도만 (무료 한도: 저장 10GiB/월 대역폭 10GiB).
- 좌표 변환 함정 (Phase 3): TUM은 Twc·(qx qy qz qw), COLMAP은 Tcw·(qw qx qy qz),
  images.txt는 이미지당 2줄, scipy quat은 (x,y,z,w). 검증은 "벽 한 장" 테스트.
