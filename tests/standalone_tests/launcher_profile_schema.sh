#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

assert_eq() {
  local expected=$1
  local actual=$2
  local label=$3
  [[ "$actual" == "$expected" ]] || {
    printf '%s: expected %q, got %q\n' "$label" "$expected" "$actual" >&2
    return 1
  }
}

reset_spec_test_state() {
  unset SPECULATIVE_METHOD SPECULATIVE_TOKENS SPECULATIVE_MODEL MTP_K
  unset VLLM_KV_CACHE_LAYOUT SPECULATIVE_ATTENTION_BACKEND
  unset SPECULATIVE_KV_CACHE_DTYPE PER_REQUEST_SPEC_DECODE_METRICS
  unset LONG_PREFILL_TOKEN_THRESHOLD
  CONFIG_OVERRIDE_SOURCE=()
  CONFIG_OVERRIDE_UNSET=()
}

reset_spec_test_state
SPECULATIVE_METHOD=none
apply_speculative_runtime_defaults
assert_eq none "$(effective_speculative_method)" "none method"
assert_eq 0 "$(effective_speculative_tokens)" "none tokens"

reset_spec_test_state
SPECULATIVE_METHOD=mtp
SPECULATIVE_MODEL=/models/retained-dflash-draft
apply_speculative_runtime_defaults
assert_eq mtp "$(effective_speculative_method)" "mtp method"
assert_eq 3 "$(effective_speculative_tokens)" "mtp default tokens"
assert_eq 3 "$MTP_K" "mtp compatibility value"
[[ -z "${VLLM_KV_CACHE_LAYOUT:-}" ]]
validate_speculative_route

VLLM_KV_CACHE_LAYOUT=custom-layout
LONG_PREFILL_TOKEN_THRESHOLD=123
apply_speculative_runtime_defaults
assert_eq custom-layout "$VLLM_KV_CACHE_LAYOUT" "custom profile KV layout"
assert_eq 123 "$LONG_PREFILL_TOKEN_THRESHOLD" "custom profile prefill threshold"

reset_spec_test_state
SPECULATIVE_METHOD=none
SPECULATIVE_MODEL=/models/retained-dflash-draft
apply_speculative_runtime_defaults
validate_speculative_route
assert_eq 0 "$(effective_speculative_tokens)" "retained draft ignored for autoregressive"

apply_profile_overrides \
  "$ROOT/profiles/2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env"
assert_eq /models/retained-dflash-draft "$SPECULATIVE_MODEL" "draft survives profile switch"
assert_eq mtp "$(effective_speculative_method)" "profile switch selects MTP"

reset_spec_test_state
SPECULATIVE_METHOD=dflash
MAX_NUM_SEQS=1
TP_SIZE=2
apply_speculative_runtime_defaults
assert_eq dflash "$(effective_speculative_method)" "dflash method"
assert_eq 7 "$(effective_speculative_tokens)" "dflash default tokens"
assert_eq BLHNC "$VLLM_KV_CACHE_LAYOUT" "dflash target KV layout"
assert_eq TRITON_ATTN "$SPECULATIVE_ATTENTION_BACKEND" "dflash attention backend"
assert_eq float16 "$SPECULATIVE_KV_CACHE_DTYPE" "dflash draft KV dtype"
assert_eq detailed "$PER_REQUEST_SPEC_DECODE_METRICS" "dflash metrics"

MODEL_DIR=/tmp/test-model
SERVED_NAME=test-model
PORT=18000
GPU_UTIL=0.9
MAX_MODEL_LEN=4096
MAX_BATCHED_TOKENS=512
MODE=fast
MODEL_FAMILY=qwen
SPECULATIVE_MODEL=/models/dflash-draft
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '--per-request-spec-decode-metrics' <<<"$args"
grep -Fxq -- 'detailed' <<<"$args"

PER_REQUEST_SPEC_DECODE_METRICS=summary
apply_speculative_runtime_defaults
validate_spec_decode_metrics
assert_eq summary "$PER_REQUEST_SPEC_DECODE_METRICS" "metrics override"

SPECULATIVE_METHOD=none
SPECULATIVE_TOKENS=0
MTP_K=0
PER_REQUEST_SPEC_DECODE_METRICS=detailed
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
! grep -Fxq -- '--per-request-spec-decode-metrics' <<<"$args"

reset_spec_test_state
SPECULATIVE_METHOD=dflash
MAX_NUM_SEQS=2
TP_SIZE=2
apply_speculative_runtime_defaults
assert_eq 2560 "$LONG_PREFILL_TOKEN_THRESHOLD" "dual-GPU DFlash C2 threshold"

reset_spec_test_state
SPECULATIVE_METHOD=dflash
MAX_NUM_SEQS=2
TP_SIZE=4
apply_speculative_runtime_defaults
assert_eq 512 "$LONG_PREFILL_TOKEN_THRESHOLD" "four-GPU DFlash C2 threshold"

allowed_keys='^(SERVED_NAME|MODE|MODEL_FAMILY|PROFILE_GROUP|MODEL_VARIANT|QUANTIZATION|KV_CACHE_DTYPE|MAX_MODEL_LEN|GPU_UTIL|MAX_BATCHED_TOKENS|MAX_NUM_SEQS|MESSAGE_TYPE|SPECULATIVE_METHOD|SPECULATIVE_TOKENS)$'
while IFS= read -r -d '' profile; do
  key_count=$(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$profile" | wc -l)
  assert_eq 14 "$key_count" "shipped profile field count: ${profile#"$ROOT/"}"
  while IFS= read -r key; do
    [[ "$key" =~ $allowed_keys ]] || {
      printf 'unexpected shipped profile key %s in %s\n' "$key" "${profile#"$ROOT/"}" >&2
      exit 1
    }
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$profile")
done < <(find "$ROOT/profiles" -type f -name '*.env' -print0 | sort -z)

echo "launcher_profile_schema_ok"
