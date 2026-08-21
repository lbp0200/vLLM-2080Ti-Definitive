# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

This directory contains launch profiles for vLLM 2080Ti Definitive. A profile
is an `.env` preset for runtime parameters only; it does not include the
checkpoint path. Choose the model directory separately in `launcher.sh` or with
`MODEL_DIR=...`.

The profiles below are carried forward from the maintained v0.1.x CUDA 12.8 /
Torch 2.11 route for launcher compatibility. Their historical throughput
figures are not v0.2.1-pre2 promotion evidence; current cu130 validation is
recorded in `docs/2080ti-0.2.1-pre-validation.md`.

This guide describes profile compatibility, not the `0.2.1-pre2` tested
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
    experimental/
      int8-w8a8/
    user/
  qwen3.8-27b/
    normal/
      fp8/
      nvfp4/
    fast/
      fp8/
      nvfp4/
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

These profile capacities and historical figures predate `0.2.1-pre2`; consult
the root tested-checkpoint list before selecting a checkpoint for cu130.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 1 | 1529.33 / 30.42 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |

### Qwen3.8 27B multimodal MTP3 (formal 0.2.1-pre2 profiles)

These routes were verified on the NVLink-connected dual RTX 2080 Ti host with
TP=2. Each route passed a natural image-answer check before measurement: the
model identified a blue square, an orange circle, and `K7P`. The retained
image-route throughput values are the maximum prefill / decode result from
three 4K-input, 128-output requests with distinct prompt prefixes. Performance
requests use `ignore_eos` only to hold a fixed 128-token decode window.
The FP8 normal row was rechecked during its 104K capacity promotion with
fixed-output synthetic 4K/128 requests; its image-answer gate is recorded
separately below.

| Checkpoint | Profile | Context | KV | Throughput |
|---|---|---:|---|---:|
| Qwen3.8-27B-FP8 | `qwen3.8-27b/normal/fp8/fp16kv-104K-mtp3-text-image.env` | 104K | FP16 | 1506.86 / 82.88 |
| Qwen3.8-27B-FP8 | `qwen3.8-27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | 240K | TQK8V4 | 1355.04 / 73.48 |
| Qwen3.8-27B-NVFP4 | `qwen3.8-27b/normal/nvfp4/fp8kv-240K-mtp3-text-image.env` | 240K | FP8 | 1266.09 / 55.69 |
| Qwen3.8-27B-NVFP4 | `qwen3.8-27b/fast/nvfp4/tqk8v4-240K-mtp3-text-image.env` | 240K | TQK8V4 | 1276.92 / 83.99 |

The FP8 normal route uses `GPU_UTIL=0.96`; startup allocated 4.04 GiB and
measured 110,784 KV tokens after CUDA Graph profiling. The 104K profile leaves
block headroom. Its three 4K/128 synthetic runs returned 128/128 tokens and
measured 1506.86 prefill / 82.88 decode tok/s (maximum of the three runs).

The normal NVFP4 route uses `compressed-tensors`, Marlin for its NVFP4 and
weight-only FP8 linear subsets on SM75, and explicit FP8 KV. Its 240K limit is
below the checkpoint's 262,144-token position limit; it is not set by KV
memory, which measured 426,080 tokens. The fast NVFP4 route measured 515,723
KV tokens but is constrained by the same checkpoint position limit. Both fast
routes use CUDA Graph decode; the NVFP4 fast route captures FULL decode plus
PIECEWISE mixed prefill/decode.

### Qwen3.8 27B FP8 text-only (formal 0.2.1-pre2 profiles)

These profiles run the official `Qwen/Qwen3.8-27B-FP8` checkpoint with
`LANGUAGE_MODEL_ONLY=1` and `SKIP_MM_PROFILING=1`. They are separate from the
multimodal profiles above and preserve CUDA Graph execution.

| Profile | Context | KV | MTP | CUDA Graph | Measured GPU KV tokens | 4K/128 prefill / decode tok/s |
|---|---:|---|---:|---|---:|---:|
| `qwen3.8-27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | 128K | FP16 | 3 | PIECEWISE, size 4 | 138,394 | 1496.95 / 83.90 |
| `qwen3.8-27b/normal/fp8/fp16kv-144K-nomtp-text-only.env` | 144K | FP16 | 0 | FULL_AND_PIECEWISE, size 1 | 152,749 | 1501.39 / 30.40 |
| `qwen3.8-27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | 256K | TQK8V4 | 3 | FULL_AND_PIECEWISE, size 4 | 310,827 | 1525.37 / 83.51 |

Each row passed the `PROFILE_OK` probe and three distinct 4K/128 synthetic
runs with complete 128-token output. Throughput is the highest valid result.

### Qwen3.8 27B NVFP4 text-only (formal 0.2.1-pre2 profiles)

These routes use `Qwen3.8-27B-NVFP4` with TP=2 and `LANGUAGE_MODEL_ONLY=1`.
Normal mode uses explicit FP8 KV; fast mode uses TurboQuant K8V4. Both passed
the `PROFILE_OK` quality probe and the fixed-token 4K/128 synthetic test with
128/128 output. Throughput is the highest valid run of three distinct prompt
variants.

| Profile | Context | KV | MTP | CUDA Graph | Measured GPU KV tokens | 4K/128 prefill / decode tok/s |
|---|---:|---|---:|---|---:|---:|
| `qwen3.8-27b/normal/nvfp4/fp8kv-240K-nomtp-text-only.env` | 240K | FP8 | 0 | PIECEWISE + FULL decode | 530,720 | 1421.10 / 38.89 |
| `qwen3.8-27b/fast/nvfp4/tqk8v4-240K-mtp3-text-only.env` | 240K | TQK8V4 | 3 | PIECEWISE + FULL decode | 564,130 | 1411.91 / 102.60 |

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

These carried-forward INT4 profile figures are not a `0.2.1-pre2`
tested-checkpoint statement.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | pending 0.2.1-pre2 rerun |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 250K per workspace | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 250K per workspace | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

### Qwen3.8 27B INT8 W8A8 (experimental deployment route)

These profiles use the measured 32K SM75 capacity envelope for
`RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP`. The 32K limit leaves headroom for
MTP3 graph and activation memory on the dual 2080 Ti host.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs |
|---|---|---:|---|---:|---|---:|
| `qwen27b/experimental/int8-w8a8/fp16kv-32K-mtp3-text-only.env` | normal | 32K | FP16 | 3 | text-only | 1 |
| `qwen27b/experimental/int8-w8a8/fp16kv-32K-nomtp-text-only.env` | normal | 32K | FP16 | 0 | text-only | 1 |
| `qwen27b/experimental/int8-w8a8/tqk8v4-32K-mtp3-text-only.env` | fast | 32K | TQK8V4 | 3 | text-only | 1 |

### Qwen3.8 27B NVFP4 (experimental ModelOpt route)

These profiles are for the `pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB` checkpoint
and must use its ModelOpt `MIXED_PRECISION` files.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs |
|---|---|---:|---|---:|---|---:|
| `qwen27b/experimental/nvfp4/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 |
| `qwen27b/experimental/nvfp4/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 |
| `qwen27b/experimental/nvfp4/tqk8v4-128K-mtp3-text-only.env` | fast | 128K | TQK8V4 | 3 | text-only | 1 |
