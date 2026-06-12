#!/usr/bin/env bash
# apply.sh — ORB_SLAM3 submodule 에 patches/NNNN-*.patch 를 번호 순서대로 적용 (idempotent).
#   fresh clone(= submodule 클린 체크아웃) 에서 실행하면 0001, 0002, ... 가 차례로 적용된다.
#   이미 적용돼 있으면 건너뛴다(reverse-check). 충돌이면 에러로 멈춘다.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="$HERE/../ORB_SLAM3"
cd "$SUB"

shopt -s nullglob
patches=( "$HERE"/[0-9]*.patch )      # 번호 prefix → glob 정렬이 곧 적용 순서
shopt -u nullglob
[ ${#patches[@]} -eq 0 ] && { echo "[apply] 적용할 패치 없음"; exit 0; }

for p in "${patches[@]}"; do
  name=$(basename "$p")
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "[apply] already applied: $name"
  elif git apply --check "$p" 2>/dev/null; then
    git apply "$p" && echo "[apply] applied: $name"
  else
    echo "[apply] ERROR: $name 적용 불가 (충돌/베이스 불일치)" >&2
    exit 1
  fi
done
echo "[apply] done (${#patches[@]} patches)"
