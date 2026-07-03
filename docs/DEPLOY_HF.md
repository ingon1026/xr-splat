# HuggingFace Spaces 데모 배포 가이드

xr-splat 웹 데모 = **Static Space**(HTML/JS만 서빙) + **dataset repo**(.splat 자산 호스팅).
렌더링은 방문자 브라우저 WebGL(gsplat.js) — 서버 GPU 불필요, 전부 무료.

```
[outputs/<scene>/web/<scene>.splat]     [spaces/index.html + README.md]
        │ (dataset repo)                        │ (Static Space)
        ▼                                       ▼
datasets/<id>/xr-splat-assets  ←─fetch──  spaces/<id>/xr-splat-demo
```

## 배포 (3단계)

```bash
# 0) 최초 1회: https://huggingface.co/settings/tokens 에서 write 토큰 발급 후
hf auth login          # (구 huggingface-cli login — hub 1.21+에서 `hf`로 개명됨)

# 1) 웹 자산 변환 (.ply → .splat, ~17MB)
python scripts/09_export_web.py --scene replica_office0

# 2) 업로드 (dataset repo + Static Space 자동 생성/갱신)
python scripts/10_deploy_hf.py --hf-id <내계정>
```

끝. 데모 URL: `https://huggingface.co/spaces/<내계정>/xr-splat-demo`

## 갱신

- **자산만 바뀜** (재학습·재변환): 09 → 10 다시 실행 (같은 파일명이라 뷰어 수정 불필요)
- **뷰어만 바뀜** (`spaces/` 수정): 10만 다시 실행
- **다른 씬 추가**: `09 --scene <s>` → `10 --scene <s>` 후 `spaces/index.html`의 `DEFAULT_URL`·오버레이 갱신

## 로컬 테스트 (배포 전)

```bash
# 레포 루트에서 — 뷰어와 .splat을 한 서버로 서빙
python -m http.server 8000
# 브라우저: http://localhost:8000/spaces/?url=/outputs/replica_office0/web/replica_office0.splat
```

`?url=` 파라미터가 기본 HF URL을 오버라이드한다 (배포본 디버깅에도 사용 가능).

## 확인 사항

- **CORS**: HF resolve 엔드포인트는 `Access-Control-Allow-Origin: *` — 배포 후 확인:
  `curl -sI '<자산 URL>' | grep -i access-control`
- **초기 카메라**: `spaces/index.html`의 `TARGET`/`RADIUS`는 09가 리포트한 값 하드코딩.
  씬이 이상한 각도로 시작하면 09 출력의 `뷰어 카메라 힌트`로 갱신.
- **.splat은 git에 커밋 금지** — `outputs/`(gitignore)에만 두고 HF Hub로 배포.

## 주의 (프라이버시)

공개 데모 씬은 **replica_office0**(공개 합성 데이터)만. home 캡처(실제 집 내부)는
09/10이 동일하게 동작하지만 **공개 업로드 금지**.
