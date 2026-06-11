# 연구 노트 (Why decoupled?)

> ⚠️ STUB — 기존 연구노트 본문 이관 대기. 아래는 SPEC §0 배경 요약이며, 상세 실험 기록은 추후 채운다.

## 전환 근거 (요약)
SplaTAM / GS-SLAM / Photo-SLAM / LoopSplat 등 **coupled**(SLAM과 Gaussian Splatting 결합) 방식을
실험한 결과 렌더링 품질·트래킹 정확도·속도가 모두 미달했다. → **역할 분리(decoupled)** 로 전환.

- **로컬라이제이션**: ORB-SLAM3 (RGB-D) — 성숙한 트래킹 + 루프 클로저 + relocalization 자산
- **실사화**: gsplat — SLAM 포즈를 **고정 입력**으로 받아 학습 (포즈 최적화 OFF + depth supervision)
- 두 맵이 동일 SLAM 포즈에서 나오므로 **좌표계 자동 공유** (별도 정합 불필요)

## TODO (이관/보강)
- [ ] coupled 각 방식의 실패 양상 정량 기록 (PSNR/ATE/속도)
- [ ] decoupled 전환 후 개선 수치
- [ ] 베이스라인 비교 실험 결과 (COLMAP 포즈 vs ORB-SLAM3 포즈, SPEC Phase 6)
