# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

This directory contains launch profiles for vLLM 2080Ti Definitive. A profile
is an `.env` preset for runtime parameters only; it does not include the
checkpoint path. Choose the model directory separately in `launcher.sh` or with
`MODEL_DIR=...`.

The profiles below are carried forward from the maintained v0.1.x CUDA 12.8 /
Torch 2.11 route for launcher compatibility. Their historical throughput
figures are not v0.2.1-pre promotion evidence; current cu130 validation is
recorded in `docs/2080ti-0.2.1-pre-validation.md`.

This guide describes profile compatibility, not the `0.2.1-pre` tested
checkpoint matrix. The intentionally limited current list is in the repository
`README.md`: Qwen3.8 27B candidate checkpoints and the retained Qwen3.x 35B
FP8 records. Gemma4 route detail is separate in `docs/gemma4-sm75-support.md`.

The shipped context and throughput numbers were validated on 2x RTX 2080 Ti
22GB cards with tensor parallel size 2.

Profile layout:

```text
profiles/
  templates/
  qwen27b/
    normal/
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
  qwen35b/
    normal/
      fp8/
    aggressive/
      fp8/
    fast/
      fp8/
    user/
```

Launch modes:

- `safe`: conservative fallback mode for maximum compatibility.
- `normal`: recommended daily mode for stable deployments.
- `fast`: high-performance mode for higher throughput.
- `aggressive`: more aggressive mode with the highest performance and quality risk.

`profiles/templates/` contains optional chat-template presets. They are selected
from the launcher as a global service setting; route profiles do not store chat
templates, GPU devices, ports, reasoning defaults, or tool-calling defaults.
For the shipped Qwen3/Qwen3.x routes, the launcher fills in `qwen3` as the
reasoning parser when one is not set so startup smoke and chat parsing stay
aligned with the model's default reasoning behavior. Set `REASONING_PARSER=off`
if you need to run without a reasoning parser for diagnostics.

File names describe the intended route:

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV positioning:

- `fp16kv`: quality route.
- `int8kv`: capacity / balance route; currently shipped only as `normal`
  profiles.
- `tqk8v4`: TurboQuant K8V4 compression route; currently shipped only for
  quality-passed `fast` profiles.
- Official Qwen3.x 35B currently ships as FP8 weight + FP16 KV presets for
  both text-only and text+image.

The shipped TQK8V4 profiles use `MAX_BATCHED_TOKENS=2560`, which is the
validated setting for the prefix-cache path with aligned Qwen hybrid cache
blocks.

## Validated Profiles

### Qwen3.x 27B FP8 (legacy capacity baseline)

These profile capacities and historical figures predate `0.2.1-pre`; consult
the root tested-checkpoint list before selecting a checkpoint for cu130.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

### Qwen3.x 35B FP8 (legacy capacity baseline)

The retained Qwen3.x 35B FP8 checkpoint records are listed in the root README.
They need independent cu130 revalidation before profile promotion.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 6705.13 / 97.33 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 1 | 5485.13 / 95.20 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | aggressive | 256K | FP16 | 0 | text-only | 1 | 6843.01 / 124.01 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | aggressive | 136K | FP16 | 0 | text+image | 1 | 5422.83 / 124.11 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | fast | 178K | FP16 | 3 | text-only | 1 | 5889.20 / 195.95 |

### Qwen3.x 27B INT4 (legacy capacity baseline)

These carried-forward INT4 profile figures are not a `0.2.1-pre`
tested-checkpoint statement.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 250K per workspace | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 250K per workspace | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |
