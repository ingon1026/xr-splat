#!/usr/bin/env bash
# ORB-SLAM3 (submodule) 재현 빌드: 패치 적용 → build.sh.
# 전제: git submodule update --init 완료. deps(Pangolin/Eigen/OpenCV 4.x)는 시스템에 설치돼 있어야 함.
# WSL2 SLAM 빌드 관례상 gcc-11 사용.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="$HERE/ORB_SLAM3"

# 1) third_party/patches/*.patch 적용 (이미 적용돼 있으면 건너뜀)
cd "$SUB"
for p in "$HERE"/patches/*.patch; do
  [ -e "$p" ] || continue
  if git apply --check "$p" 2>/dev/null; then
    git apply "$p" && echo "[patch] applied $(basename "$p")"
  else
    echo "[patch] skip (already applied or N/A): $(basename "$p")"
  fi
done

# 2) 빌드 (gcc-11)
if [ -x /usr/bin/gcc-11 ]; then export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11; fi
echo "[build] CC=${CC:-default}"
chmod +x build.sh
./build.sh
echo "[build] done -> lib/libORB_SLAM3.so, Examples/RGB-D/rgbd_tum"
