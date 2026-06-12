#!/usr/bin/env bash
# ORB-SLAM3 (submodule) 재현 빌드: 패치 적용 → build.sh.
# 전제: git submodule update --init 완료. deps(Pangolin/Eigen/OpenCV 4.x)는 시스템에 설치돼 있어야 함.
# WSL2 SLAM 빌드 관례상 gcc-11 사용.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="$HERE/ORB_SLAM3"

# 1) 패치 적용 (번호 순서 idempotent)
bash "$HERE/patches/apply.sh"

# 2) 빌드 (gcc-11)
cd "$SUB"
if [ -x /usr/bin/gcc-11 ]; then export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11; fi
echo "[build] CC=${CC:-default}"
chmod +x build.sh
./build.sh
echo "[build] done -> lib/libORB_SLAM3.so, Examples/RGB-D/rgbd_tum"
