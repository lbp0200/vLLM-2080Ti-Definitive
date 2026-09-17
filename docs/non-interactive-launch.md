# Non-Interactive Launch

[简体中文](non-interactive-launch.zh-CN.md)

Non-interactive mode starts a configured vLLM service without opening the
launcher menu. Build the runtime first with `./build.sh`.

## Start With A Profile

Pass the model directory, a profile path relative to `profiles/`, the target
GPUs, and the matching launch mode:

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env \
  --mode normal \
  --gpu-devices 4,5 \
  --tp-size 2 \
  --pp-size 1
```

Any recognized configuration option enables non-interactive mode, so
`--non-interactive` is optional in this form. The launcher waits for `/health`,
runs a smoke request, and prints the API URL, PID file, and log file.

## Preview Before Starting

Add `--print-config` to resolve the profile and all overrides without starting
the service:

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env \
  --mode normal \
  --gpu-devices 4,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

## Common Options

| Option | Purpose | Example |
| --- | --- | --- |
| `--model-dir` | Checkpoint directory | `/mnt/models/Qwen3.8-27B-FP8` |
| `--profile` | Route preset relative to `profiles/` | `2x2080Ti/qwen27b/w8a16/...env` |
| `--mode` | `safe`, `normal`, `fast`, or `aggressive` | `fast` |
| `--gpu-devices` | Physical GPU IDs | `4,5` |
| `--tp-size` / `--pp-size` | TP/PP layout | `2` / `1` |
| `--port` | API port | `8000` |
| `--service-scope` | `local` or `lan` | `local` |
| `--start-timeout` | Startup timeout in seconds | `900` |
| `--print-config` | Print the resolved configuration and exit | no value |

Profile values can be overridden with their lower-kebab-case option. For
advanced launcher or `VLLM_*` settings, use `--set KEY=VALUE`; use
`--unset KEY` to discard an inherited value and return to the launcher default.

```bash
./launcher.sh \
  --model-dir /path/to/checkpoint \
  --profile 2x2080Ti/qwen27b/w4a16/fast/dflash2-tqk8v4-1x256k-text-image.env \
  --mode fast \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --port 18080 \
  --set HF_DOWNLOAD_ROUTE_MODE=official
```

Configuration precedence is `CLI > exported environment > profile > default`.
Boolean options accept `0/1`; a boolean flag without a value means `1`.

## Environment Variable Form

The equivalent environment-based form is also supported:

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env \
MODE=normal \
GPU_DEVICES=4,5 \
TP_SIZE=2 \
PP_SIZE=1 \
NON_INTERACTIVE=1 \
./launcher.sh
```

Use CLI options for automation when possible because the effective overrides
are visible in the command itself.

## Logs And Stopping

By default, logs, PID files, and launcher state are written under `run-logs/`.
The launcher prints the exact paths after a successful start. To stop a managed
service, run `./launcher.sh` and choose **Stop service**; there is currently no
non-interactive stop option.
