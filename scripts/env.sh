#!/usr/bin/env bash
# Common local environment shim for xr-splat scripts.
#
# Override from the shell when your paths differ:
#   XRSPLAT_CONDA_SH=/path/to/conda.sh XRSPLAT_ENV=xrsplat bash scripts/06_train_gsplat.sh scene
#   COLMAP_BIN=/path/to/colmap COLMAP_LIB_DIR=/path/to/lib bash scripts/localize_query_colmap.sh

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

XRSPLAT_ENV="${XRSPLAT_ENV:-xrsplat}"
XRSPLAT_CONDA_SH="${XRSPLAT_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [ -f "$XRSPLAT_CONDA_SH" ]; then
  # conda activate may touch unset shell vars.
  # shellcheck disable=SC1090
  source "$XRSPLAT_CONDA_SH"
  set +u 2>/dev/null || true
  conda activate "$XRSPLAT_ENV"
  set -u 2>/dev/null || true
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [ -x /usr/bin/gcc-11 ] && [ -x /usr/bin/g++-11 ]; then
  export CC="${CC:-/usr/bin/gcc-11}"
  export CXX="${CXX:-/usr/bin/g++-11}"
fi

COLMAP_BIN="${COLMAP_BIN:-}"
if [ -z "$COLMAP_BIN" ]; then
  if command -v colmap >/dev/null 2>&1; then
    COLMAP_BIN="$(command -v colmap)"
  elif [ -x "$HOME/miniconda3/envs/colmap/bin/colmap" ]; then
    COLMAP_BIN="$HOME/miniconda3/envs/colmap/bin/colmap"
  fi
fi

COLMAP_LIB_DIR="${COLMAP_LIB_DIR:-}"
if [ -z "$COLMAP_LIB_DIR" ] && [ -d "$HOME/miniconda3/envs/colmap/lib" ]; then
  COLMAP_LIB_DIR="$HOME/miniconda3/envs/colmap/lib"
fi
if [ -n "$COLMAP_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$COLMAP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ -z "${XRSPLAT_PYTHON:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    XRSPLAT_PYTHON="$CONDA_PREFIX/bin/python"
  else
    XRSPLAT_PYTHON="$(command -v python3)"
  fi
fi

export ROOT XRSPLAT_ENV XRSPLAT_CONDA_SH COLMAP_BIN COLMAP_LIB_DIR XRSPLAT_PYTHON
