#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"

LOG_DIR=${BUILD_LOG_DIR:-"$ROOT/build-logs"}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/build-$STAMP.log"
exec > >(tee -a "$LOG_FILE") 2>&1

fail() {
  echo "BUILD FAILED: $*" >&2
  echo "Log: $LOG_FILE" >&2
  exit 1
}

is_positive_integer() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

cuda_backend="cu$(printf '%s' "$PRIMARY_CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
max_jobs=${MAX_JOBS:-24}
allow_host_mismatch=${ALLOW_HOST_MISMATCH:-0}
require_primary_env=${REQUIRE_PRIMARY_ENV:-1}
python_version=${PYTHON_VERSION:-$PRIMARY_PYTHON_VERSION}
venv_dir=${VENV_DIR:-"$ROOT/.venv"}
python_bin="$venv_dir/bin/python"

is_positive_integer "$max_jobs" || fail "MAX_JOBS must be a positive integer."
require_command uv

if [[ ! -f pyproject.toml || ! -d vllm ]]; then
  fail "Run this script from the 0.2.1 source tree."
fi

check_primary_host() {
  local failures=()
  local nvcc_major gcc_major kernel_major

  if command -v nvcc >/dev/null 2>&1; then
    nvcc_major=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)
    if [[ "$nvcc_major" != "${PRIMARY_CUDA_VERSION%%.*}" ]]; then
      failures+=("nvcc major ${nvcc_major:-unknown} (expected ${PRIMARY_CUDA_VERSION%%.*})")
    fi
  else
    failures+=("nvcc is not on PATH")
  fi

  if command -v gcc >/dev/null 2>&1; then
    gcc_major=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$gcc_major" != "$PRIMARY_GCC_MAJOR" ]]; then
      failures+=("GCC major ${gcc_major:-unknown} (expected $PRIMARY_GCC_MAJOR)")
    fi
  else
    failures+=("gcc is not on PATH")
  fi

  kernel_major=$(uname -r | cut -d. -f1)
  if ! is_positive_integer "$kernel_major" || (( kernel_major < PRIMARY_MIN_KERNEL_MAJOR )); then
    failures+=("Linux kernel ${kernel_major:-unknown} (expected >= $PRIMARY_MIN_KERNEL_MAJOR)")
  fi

  if ((${#failures[@]} > 0)); then
    printf 'Primary host checks: %s\n' "${failures[*]}"
    if [[ "$require_primary_env" == "1" && "$allow_host_mismatch" != "1" ]]; then
      fail "Host does not match the 0.2.1 target. Set ALLOW_HOST_MISMATCH=1 only for a non-target dry run."
    fi
    echo "Continuing because ALLOW_HOST_MISMATCH=1 or REQUIRE_PRIMARY_ENV=0."
  else
    echo "Primary host checks: CUDA ${PRIMARY_CUDA_VERSION}, GCC ${PRIMARY_GCC_MAJOR}, kernel >= ${PRIMARY_MIN_KERNEL_MAJOR}"
  fi
}

echo "============================================================"
echo "vLLM 2080 Ti Definitive Edition ${FORK_RELEASE} source build"
echo "Upstream base: vLLM ${BASE_VLLM_VERSION}"
echo "Target: CUDA ${PRIMARY_CUDA_VERSION} / Torch ${PRIMARY_TORCH_VERSION} / SM75"
echo "Build log: $LOG_FILE"
echo "============================================================"

check_primary_host

export MAX_JOBS="$max_jobs"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-$cuda_backend}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

if [[ ! -x "$python_bin" ]]; then
  echo "Creating Python ${python_version} environment at $venv_dir"
  uv venv --python "$python_version" "$venv_dir"
fi

[[ -x "$python_bin" ]] || fail "Python environment was not created: $python_bin"

echo "Installing and compiling vLLM with uv torch backend $cuda_backend"
uv pip install --python "$python_bin" -e "." "--torch-backend=$cuda_backend"

echo "Checking the resulting runtime"
EXPECTED_TORCH_VERSION="$PRIMARY_TORCH_VERSION" \
EXPECTED_CUDA_BACKEND="$cuda_backend" \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  "$python_bin" - <<'PY'
import os

import torch
import vllm

expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
expected_backend = os.environ["EXPECTED_CUDA_BACKEND"]
if not torch.__version__.startswith(expected_torch):
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda is None:
    raise SystemExit("the installed torch build has no CUDA runtime")

print(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")
print(f"Torch version: {torch.__version__}")
print(f"Torch CUDA runtime: {torch.version.cuda} (requested {expected_backend})")
print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: capability={torch.cuda.get_device_capability(index)}")
PY

echo "BUILD SUCCEEDED"
echo "Runtime: $python_bin"
