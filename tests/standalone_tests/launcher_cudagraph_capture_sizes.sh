#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

assert_capture_sizes() {
  local speculative_tokens=$1
  local max_num_seqs=$2
  local expected=$3
  local actual

  actual=$(speculative_cudagraph_capture_sizes \
    "$speculative_tokens" "$max_num_seqs")
  [[ "$actual" == "$expected" ]] || {
    printf 'expected capture sizes %s for K=%s C=%s\n' \
      "$expected" "$speculative_tokens" "$max_num_seqs" >&2
    printf 'actual: %s\n' "$actual" >&2
    return 1
  }
}

assert_capture_sizes 7 1 "8"
assert_capture_sizes 7 2 "8,16"
assert_capture_sizes 7 3 "8,16,24"
assert_capture_sizes 5 3 "6,12,18"

unset LONG_PREFILL_TOKEN_THRESHOLD PREFILL_BATCH_BARRIER TP_SIZE
apply_profile_overrides \
  "$ROOT/profiles/2x2080Ti/qwen27b/w4a16/fast/dflash2-tqk8v4-2x172k-text-only.env"
TP_SIZE=2
apply_speculative_runtime_defaults
[[ "$LONG_PREFILL_TOKEN_THRESHOLD" == "2560" ]]
[[ -z "${PREFILL_BATCH_BARRIER:-}" ]]

MODEL_DIR=/tmp/test-model
SERVED_NAME=test-model
PORT=18000
GPU_UTIL=0.9
MAX_MODEL_LEN=4096
MAX_BATCHED_TOKENS=512
MAX_NUM_SEQS=3
TP_SIZE=2
MODE=fast
MODEL_FAMILY=qwen
SPECULATIVE_CONFIG='{"method":"dflash","model":"draft","num_speculative_tokens":7}'
LONG_PREFILL_TOKEN_THRESHOLD=256
VLLM_ARGS=()

configure_automatic_prefill_batch_barrier
[[ "$PREFILL_BATCH_BARRIER" == "1" ]]

build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '--prefill-batch-barrier' <<<"$args"
grep -Fxq -- '--long-prefill-token-threshold' <<<"$args"
grep -Fxq -- '256' <<<"$args"
grep -Fxq -- '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16,24],"max_cudagraph_capture_size":24}' <<<"$args"

MAX_NUM_SEQS=1
configure_automatic_prefill_batch_barrier
[[ "$PREFILL_BATCH_BARRIER" == "0" ]]
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
! grep -Fxq -- '--prefill-batch-barrier' <<<"$args"

echo "launcher_cudagraph_capture_sizes_ok"
