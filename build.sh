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

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-13.0 /usr/local/cuda-13; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_HOME="$candidate"
      export CUDA_HOME
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/bin" ]]; then
  case ":$PATH:" in
    *:"$CUDA_HOME/bin":*) ;;
    *) PATH="$CUDA_HOME/bin:$PATH"; export PATH ;;
  esac
fi
if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME:-}/.local/bin/uv" ]]; then
  PATH="${HOME}/.local/bin:$PATH"
  export PATH
fi

# Prefer prepared source trees when a previous build populated .deps. This
# avoids repeating large Git clones while keeping the upstream FetchContent
# defaults for a clean checkout.
if [[ -z "${TRITON_KERNELS_SRC_DIR:-}" ]]; then
  for candidate in \
    "$ROOT/.deps/triton_kernels-src/python/triton_kernels/triton_kernels" \
    "$ROOT/.deps/triton-kernels-tar"/triton-*/python/triton_kernels/triton_kernels; do
    if [[ -d "$candidate" ]]; then
      export TRITON_KERNELS_SRC_DIR="$candidate"
      break
    fi
  done
fi
if [[ -z "${VLLM_CUTLASS_SRC_DIR:-}" ]]; then
  for candidate in \
    "$ROOT/.deps/cutlass-src" \
    "$ROOT/.deps/cutlass-tar"/cutlass-*/; do
    if [[ -f "$candidate/include/cutlass/cutlass.h" ]]; then
      export VLLM_CUTLASS_SRC_DIR="$candidate"
      break
    fi
  done
fi

is_positive_integer() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

detect_cpu_threads() {
  local threads
  threads=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if ! is_positive_integer "$threads"; then
    threads=$(nproc 2>/dev/null || true)
  fi
  if ! is_positive_integer "$threads"; then
    threads=4
  fi
  echo "$threads"
}

detect_memory_gb() {
  local mem_kb mem_gb
  mem_kb=$(awk '/MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null || echo 0)
  mem_gb=$(( mem_kb / 1024 / 1024 ))
  if (( mem_gb < 1 )); then
    mem_gb=1
  fi
  echo "$mem_gb"
}

select_max_jobs() {
  local threads=$1
  local mem_gb=$2
  local cap=${3:-}
  local reserve_gb=3
  local per_job_gb=3
  local mem_limited_jobs

  mem_limited_jobs=$(( (mem_gb - reserve_gb) / per_job_gb ))
  (( mem_limited_jobs >= 1 )) || mem_limited_jobs=1
  if [[ -n "$cap" ]]; then
    (( mem_limited_jobs <= cap )) || mem_limited_jobs=$cap
  fi
  (( mem_limited_jobs <= threads )) || mem_limited_jobs=$threads

  echo "$mem_limited_jobs"
}

validate_max_jobs_range() {
  local jobs=$1
  local threads=$2
  is_positive_integer "$jobs" || fail "MAX_JOBS must be a positive integer."
  if (( jobs < 1 || jobs > threads )); then
    fail "MAX_JOBS must be between 1 and CPU_THREADS ($threads)."
  fi
}

cuda_backend="cu$(printf '%s' "$PRIMARY_CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
cpu_threads=${CPU_THREADS:-$(detect_cpu_threads)}
memory_gb=${MEMORY_GB:-$(detect_memory_gb)}
auto_max_jobs_cap=${BUILD_AUTO_MAX_JOBS_CAP:-8}
is_positive_integer "$cpu_threads" || fail "CPU_THREADS must be a positive integer when set explicitly."
is_positive_integer "$memory_gb" || fail "MEMORY_GB must be a positive integer when set explicitly."
is_positive_integer "$auto_max_jobs_cap" || fail "BUILD_AUTO_MAX_JOBS_CAP must be a positive integer when set explicitly."

if [[ -n "${BUILD_MAX_JOBS:-}" && -n "${MAX_JOBS:-}" && "${BUILD_MAX_JOBS}" != "${MAX_JOBS}" ]]; then
  fail "BUILD_MAX_JOBS and MAX_JOBS must match when both are set."
fi
if [[ -n "${BUILD_MAX_JOBS:-}" ]]; then
  max_jobs=$BUILD_MAX_JOBS
  max_jobs_source=env:BUILD_MAX_JOBS
elif [[ -n "${MAX_JOBS:-}" ]]; then
  max_jobs=$MAX_JOBS
  max_jobs_source=env:MAX_JOBS
else
  auto_jobs_without_cap=$(select_max_jobs "$cpu_threads" "$memory_gb" "$cpu_threads")
  max_jobs=$(select_max_jobs "$cpu_threads" "$memory_gb" "$auto_max_jobs_cap")
  if (( max_jobs < auto_jobs_without_cap )); then
    max_jobs_source=auto-cap
  elif (( max_jobs < cpu_threads )); then
    max_jobs_source=auto-memory
  else
    max_jobs_source=auto
  fi
fi
allow_host_mismatch=${ALLOW_HOST_MISMATCH:-0}
require_primary_env=${REQUIRE_PRIMARY_ENV:-1}
python_version=${PYTHON_VERSION:-$PRIMARY_PYTHON_VERSION}
venv_dir=${VENV_DIR:-"$ROOT/.venv"}
python_bin="$venv_dir/bin/python"
git_mirror_prefix=${BUILD_GIT_MIRROR_PREFIX:-}
flashqla_repo=${FLASHQLA_REPO:-https://github.com/weicj/FlashQLA-SM70-SM75.git}
flashqla_dir=${FLASHQLA_DIR:-"$ROOT/.deps/FlashQLA-SM70-SM75"}
flashqla_enabled=${FLASHQLA_ENABLED:-1}
flashqla_clone_timeout=${FLASHQLA_CLONE_TIMEOUT:-180}
skip_vllm_build=${SKIP_VLLM_BUILD:-0}

validate_max_jobs_range "$max_jobs" "$cpu_threads"
is_positive_integer "$flashqla_clone_timeout" || fail "FLASHQLA_CLONE_TIMEOUT must be a positive integer."
[[ "$skip_vllm_build" == "0" || "$skip_vllm_build" == "1" ]] ||
  fail "SKIP_VLLM_BUILD must be 0 or 1."
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

check_cuda_glibc_compatibility() {
  local header="$CUDA_HOME/targets/x86_64-linux/include/crt/math_functions.h"
  local glibc_version

  glibc_version=$(ldd --version 2>/dev/null | awk 'NR == 1 { print $NF }')
  [[ "$glibc_version" =~ ^[0-9]+\.[0-9]+$ ]] || return 0
  if [[ "$(printf '%s\n%s\n' 2.41 "$glibc_version" | sort -V | head -n 1)" != "2.41" ]]; then
    return 0
  fi

  if ! grep -q '__GLIBC_PREREQ(2,41)' "$header" || \
    ! grep -q '_NV_RSQRT_SPECIFIER' "$header"; then
    fail "CUDA $PRIMARY_CUDA_VERSION requires toolchain-patches/cuda-13.0-glibc-2.41-rsqrt.patch on glibc $glibc_version. Apply it to CUDA_HOME before rebuilding."
  fi
}

prepare_flashqla_sm75() {
  [[ "$flashqla_enabled" == "1" ]] || {
    echo "FlashQLA SM70/SM75 backend: disabled (FLASHQLA_ENABLED=$flashqla_enabled)"
    return 0
  }

  mkdir -p "$(dirname -- "$flashqla_dir")"
  if [[ ! -e "$flashqla_dir" ]]; then
    echo "Fetching FlashQLA SM70/SM75 backend from $flashqla_repo"
    clone_flashqla() {
      local source=$1
      if command -v timeout >/dev/null 2>&1; then
        timeout --foreground "$flashqla_clone_timeout" \
          git clone --depth=1 "$source" "$flashqla_dir"
      else
        git clone --depth=1 "$source" "$flashqla_dir"
      fi
    }
    if [[ -n "$git_mirror_prefix" ]]; then
      clone_flashqla "${git_mirror_prefix}${flashqla_repo#https://github.com/}" || {
        rm -rf "$flashqla_dir"
        clone_flashqla "$flashqla_repo"
      }
    else
      clone_flashqla "$flashqla_repo"
    fi
  fi

  [[ -f "$flashqla_dir/flash_qla/ops/gated_delta_rule/legacy/sm_legacy.py" ]] ||
    fail "FlashQLA checkout is missing the SM70/SM75 legacy backend"

  "$python_bin" tools/patch_flashqla_sm75_imports.py \
    "$flashqla_dir" "$ROOT/tools/flashqla_sm75_patches"

  echo "Installing FlashQLA Python package into $venv_dir"
  uv pip install --python "$python_bin" --no-deps -e "$flashqla_dir"

  local extension_dir="${TORCH_EXTENSIONS_DIR:-$flashqla_dir/.torch_extensions_vllm_flashqla_legacy}"
  echo "Building FlashQLA SM75 legacy extension in $extension_dir"
  CUDA_HOME="$CUDA_HOME" \
  CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}" \
  CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  TORCH_EXTENSIONS_DIR="$extension_dir" \
  PYTHONPATH="$flashqla_dir${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" - <<'PY'
from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import _load_ext

extension = _load_ext()
required = ("gdn_forward", "gdn_forward_varlen")
missing = [name for name in required if not hasattr(extension, name)]
if missing:
    raise SystemExit(
        "FlashQLA extension is stale; missing exported symbols: "
        + ", ".join(missing)
        + ". Re-run the build after applying the SM75 packed-varlen patch."
    )
print(f"FlashQLA extension: {extension.__file__}")
print("FlashQLA symbols: " + ", ".join(required))
PY

  export FLASHQLA_DIR="$flashqla_dir"
  export TORCH_EXTENSIONS_DIR="$extension_dir"
  export PYTHONPATH="$flashqla_dir${PYTHONPATH:+:$PYTHONPATH}"
}

echo "============================================================"
echo "vLLM 2080 Ti Definitive Edition ${FORK_RELEASE} source build"
echo "Upstream base: vLLM ${BASE_VLLM_VERSION}"
echo "Target: CUDA ${PRIMARY_CUDA_VERSION} / Torch ${PRIMARY_TORCH_VERSION} / SM75"
echo "Build parallelism: ${max_jobs} (${max_jobs_source}; CPU=${cpu_threads}, memory=${memory_gb}GiB, auto cap=${auto_max_jobs_cap})"
echo "Build log: $LOG_FILE"
echo "============================================================"

check_primary_host
check_cuda_glibc_compatibility

export MAX_JOBS="$max_jobs"
export BUILD_MAX_JOBS=${BUILD_MAX_JOBS:-$MAX_JOBS}
export CPU_THREADS="$cpu_threads"
export MEMORY_GB="$memory_gb"
export AUTO_MAX_JOBS_CAP="$auto_max_jobs_cap"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-$cuda_backend}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

if [[ -n "$git_mirror_prefix" ]]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${git_mirror_prefix}https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
fi

# A source snapshot made with git archive, or a copied worktree whose `.git`
# file points at unavailable metadata, has no usable SCM information. Keep
# the package version deterministic in both supported validation layouts.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-$BASE_VLLM_VERSION}"
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-$BASE_VLLM_VERSION}"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Creating Python ${python_version} environment at $venv_dir"
  uv venv --python "$python_version" "$venv_dir"
fi

[[ -x "$python_bin" ]] || fail "Python environment was not created: $python_bin"

# Build helpers such as ninja are installed into the venv by uv. Keep the
# editable FlashQLA build and the vLLM build on the same toolchain PATH.
case ":$PATH:" in
  *:"$venv_dir/bin":*) ;;
  *) PATH="$venv_dir/bin:$PATH"; export PATH ;;
esac

if [[ "$skip_vllm_build" == "1" ]]; then
  echo "Skipping vLLM build (SKIP_VLLM_BUILD=1); validating existing editable install"
else
  echo "Installing and compiling vLLM with uv torch backend $cuda_backend"
  uv pip install --python "$python_bin" -e "." "--torch-backend=$cuda_backend"
fi

prepare_flashqla_sm75

echo "Checking the resulting runtime"
EXPECTED_TORCH_VERSION="$PRIMARY_TORCH_VERSION" \
EXPECTED_CUDA_BACKEND="$cuda_backend" \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
FLASHQLA_ENABLED="$flashqla_enabled" \
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
if os.environ.get("FLASHQLA_ENABLED") == "1":
    from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import _load_ext

    extension = _load_ext()
    required = ("gdn_forward", "gdn_forward_varlen")
    missing = [name for name in required if not hasattr(extension, name)]
    if missing:
        raise SystemExit(
            "FlashQLA extension is stale; missing exported symbols: "
            + ", ".join(missing)
        )
    print(f"FlashQLA extension: {extension.__file__}")
    print("FlashQLA symbols: " + ", ".join(required))
PY

echo "BUILD SUCCEEDED"
echo "Runtime: $python_bin"
