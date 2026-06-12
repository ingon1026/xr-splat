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
  - RealSense 라이브 스트리밍 WSL2 비신뢰 → Windows RealSense Viewer로 .bag 녹화 (Plan A)
  - COLMAP 베이스라인: `run_colmap_sfm.sh`(고정 TUM intrinsics, 127/127 등록) + `setup_sfm_baseline.py`(metric 스케일정렬 **s=0.234**, IQR/median 0.03)
  - 08 후처리: opacity prune→SH 3→1→경량, **421MB→105MB(-0.48dB)**
- 다음: **M3** (README Results 표를 `outputs/tum_fr1_desk/eval_m1.json`으로 채우기 + 티저 GIF/이미지). **M2**는 내 D455 .bag 캡처 대기.
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
