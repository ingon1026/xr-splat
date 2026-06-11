# CLAUDE.md — XR-Splat 작업 규칙

이 저장소에서 Claude Code가 따르는 작업 규칙. SPEC과 충돌 시 아래 1번을 따른다.

## 1. 단일 진실 공급원
- `SPEC.md`가 단일 진실 공급원이다. 명세와 충돌하는 내용이 있으면 **항상 SPEC을 우선**한다.
- 아키텍처 결정(decoupled SLAM+GS, ORB-SLAM3, gsplat, 포즈 고정)은 변경 금지·대안 제시 금지.

## 2. Phase 순서 엄수 (게이트)
- Phase는 순서대로 진행한다. 각 Phase의 **Acceptance Criteria를 실제 실행 증거(로그/출력)와 함께 통과**하기 전에는 다음 Phase로 넘어가지 않는다.
- **특히 `05_validate_poses.py`가 PASS 하기 전에는 gsplat 학습을 절대 시작하지 않는다.**

## 3. 블로킹 금지 (백그라운드 필수)
- **5분 이상 걸리는 작업**(ORB-SLAM3 빌드, gsplat 학습, .bag 추출 등)은 **절대 포그라운드로 돌리지 않는다.**
- 반드시 백그라운드로 실행한다(`nohup` + 로그파일). 완료를 기다리지 말고, 그동안 다음 Phase 코드 작업을 진행한다.
- 상태는 **로그 tail로만** 확인한다. 작업이 끝나면 로그를 근거로 성공/실패를 판정해 보고한다.

## 4. ORB-SLAM3 실행 (WSLg)
- **ORB-SLAM3는 항상 뷰어 OFF(headless)로 실행한다.** WSLg에서 Pangolin 뷰어가 OpenGL(ZINK/EGL)로 깨지고, 트래젝토리 저장 후 셧다운에서 segfault가 나 종료코드가 0이 아니기 때문이다.
- **성공 판정은 종료코드가 아니라 `KeyFrameTrajectory.txt` 생성(과 내용 유효성) 여부로 한다.**

## 5. 검증 (적대적 리뷰)
- **Phase 3 완료 시**, 서브에이전트를 띄워 `03_tum_to_colmap.py`를 SPEC의 **함정 체크리스트 4개** 기준으로 적대적 리뷰시킨 뒤 커밋한다.

## 6. 커밋 & 결정
- 커밋은 **Phase 단위 conventional commits**(`feat:`, `fix:`, `docs:`, `test:`)로 만든다.
- **막히는 결정(라이브러리 선택, 명세 모호함)은 임의로 정하지 않고 사용자에게 질문**한다.
