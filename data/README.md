# 데이터 안내

이 디렉토리의 `raw/`, `processed/`는 git에서 제외된다 (SPEC §5.1). 구조만 `.gitkeep`으로 유지.
아래 방법으로 데이터를 직접 입수한다.

## 디렉토리 규약
```
data/
├── raw/          # .bag 원본 (자체 캡처) — gitignore
└── processed/    # TUM RGB-D, COLMAP 포맷 (스크립트 산출) — gitignore
    └── <scene>/  # 장면 단위
```

## 공개 데이터셋 (먼저 이걸로 재현 — M1)
GT 궤적이 있어 "촬영 문제"와 "파이프라인 문제"를 분리해준다. 자체 캡처(M2)보다 **먼저** 통과시킨다.

- **TUM RGB-D**: `fr1/desk` 등. https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
  - depth PNG 스케일 = 5000 (DepthMapFactor)
- **Replica (SLAM)**: 합성 시퀀스. depth 스케일 ≈ 6553.5
- 공개 데이터는 `.bag`이 없으므로 `01_extract_bag.py`의 **디렉토리 입력 모드**로 인제스트한다
  (`intrinsics.json` / `associations.txt` / `DepthMapFactor` 합성).

## 자체 캡처 (D455 — M2)
`docs/capture-guide.md` 참고. 녹화한 `.bag`을 `data/raw/<scene>.bag`에 둔다 (depth 스케일 1000).
