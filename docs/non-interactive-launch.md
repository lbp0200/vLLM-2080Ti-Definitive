# Non-Interactive Launch

Build the runtime with `./build.sh`, then pass a flat profile path. Profile
files contain route parameters only; checkpoint, GPU topology, port, and mode
remain launcher options.

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` is optional and defaults to `fast`. A profile without `MODE` keeps
the launcher selection; an explicit profile `MODE=normal` or `MODE=fast` may
override it. There are no `fast/` or `normal/` profile directories.

Useful options include `--model-dir`, `--speculative-model`, `--profile`,
`--mode`, `--gpu-devices`, `--tp-size`, `--pp-size`, `--port`,
`--start-timeout`, and `--print-config`. Use `--set KEY=VALUE` for advanced
launcher/runtime settings. Do not add Prefix Cache, Mamba cache, GPU, port, or
model-path fields to a profile; the validator rejects them.

The profile library is a validation matrix, not a promise that every filename
fits every machine. Capacity and performance are promoted only after the
external audit records startup, 4K/128, 32K/512, concurrency, and image
correctness evidence.
