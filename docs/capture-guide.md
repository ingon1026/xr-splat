# D455 캡처 가이드

오프라인 처리 파이프라인이므로 라이브 스트리밍은 불필요하다. Windows에서 녹화한 `.bag`을
WSL2로 옮겨 처리한다 (SPEC §1.1 Plan A).

## 캡처 절차 (Plan A)
1. Windows에서 **Intel RealSense Viewer**로 녹화 → `.bag` 생성
2. `.bag`을 WSL2 **내부 디스크**로 복사 (`/mnt/c/...` 경유보다 I/O 빠름)
3. `01_extract_bag.py`로 오프라인 파싱 (카메라 불필요, `rs.config.enable_device_from_file()`)

## 권장 캡처 설정 (SPEC §1.2)
- 해상도: **color 1280x720 @ 30fps**, **depth 848x480 @ 30fps** (D455 depth 최적은 848x480)
- **auto-exposure / auto-white-balance OFF** (수동 고정) — 프레임 간 색 일관성
- depth↔color 해상도가 다르므로 **align(depth→color)** 필수 (`rs.align`)
- depth scale: D455 기본 1mm (`depth_scale=0.001`). 코드 하드코딩 금지, SDK에서 읽을 것

## 촬영 요령
- 천천히 이동, 같은 영역을 다각도로
- **시작 지점으로 복귀**하여 루프 클로저 유도
