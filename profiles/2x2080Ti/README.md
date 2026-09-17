# 2x2080Ti Profiles

Hardware-specific profiles for two RTX 2080 Ti GPUs. Available model families
are Qwen 27B NVFP4 (`qwen27b/w4a16`), Qwen 27B FP8 (`qwen27b/w8a16`), and Qwen
35B FP8 (`qwen35b/w8a16`).

Validated DFlash2 routes:

| Profile | Mode | KV | Concurrency/context | Messages |
|---|---|---|---|---|
| `qwen27b/w4a16/fast/dflash2-tqk8v4-2x172k-text-only.env` | fast | TQK8V4 | 2 x 172K | text-only |
| `qwen27b/w4a16/fast/dflash2-tqk8v4-1x256k-text-image.env` | fast | TQK8V4 | 1 x 256K | text+image |

Other measured routes remain under their model/weight directories.

## Qwen3.8-27B-FP8

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w8a16/normal/mtp-fp16kv-1x104k-text-image.env` | normal | 104K | FP16 | 3 | text+image | 110,784 | 1506.86 / 82.88 |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x128k-text-only.env` | normal | 128K | FP16 | 3 | text-only | 138,394 | 1496.95 / 83.90 |
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x144k-text-only.env` | normal | 144K | FP16 | 0 | text-only | 152,749 | 1501.39 / 30.40 |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 310,827 | 1525.37 / 83.51 |
| `qwen27b/w8a16/normal/mtp-fp8kv-1x220k-text-image.env` | normal | 220K | FP8 | 3 | text+image | 231,169 | 1555.1 / 70.1 |
| `qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env` | normal | 256K | FP8 | 3 | text-only | 320,232 | 1573.95 / 67.58 |

## Qwen3.8-27B-NVFP4

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w4a16/fast/dflash2-tqk8v4-2x172k-text-only.env` | fast | 172K x2 | TQK8V4 | DFlash2/7 | text-only | 369,439 | ~1400 / 206 |
| `qwen27b/w4a16/fast/dflash2-tqk8v4-1x256k-text-image.env` | fast | 256K x1 | TQK8V4 | DFlash2/7 | text+image | 318,010 | 1445.2 / 208.7 |
| `qwen27b/w4a16/normal/mtp-fp8kv-1x240k-text-only.env` | normal | 240K | FP8 | 3 | text-only | 463,890 | 1433.2 / 76.8 |
| `qwen27b/w4a16/normal/mtp-fp8kv-1x240k-text-image.env` | normal | 240K | FP8 | 3 | text+image | 426,080 | 1250.6 / 52.5 |
| `qwen27b/w4a16/normal/nomtp-fp8kv-8x192k-text-only.env` | normal | 192K | FP8 | 0 | text-only | 518,191 | 1372.1 / 42.0 |
| `qwen27b/w4a16/fast/mtp-tq4nc-8x262k-text-only.env` | fast | 262K | TQ4NC | 3 | text-only | 732,381 | 1402.9 / 103.5 |

DFlash2's warmed 4K/128 standard Launcher request measured approximately
**1,445 / 209 tok/s** (prefill / decode) on the 1x256K route; the first request
is excluded because it may trigger rejection and TurboQuant kernel JIT.

### Concurrent benchmark lanes

| Profile | Mode | Context | KV/MTP | GPU KV tokens | C1 | C2 | C4 | C8 | Evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `qwen27b/w4a16/normal/nomtp-fp8kv-8x192k-text-only.env` | normal | 192K | FP8 / 0 | 518,191 | 1372.1 / 42.0 | 1507.4 / 80.4 | 1535.9 / 152.0 | 1523.4 / 270.8 | full-window run |
| `qwen27b/w4a16/fast/mtp-tq4nc-8x262k-text-only.env` | fast | 262K | TQ4NC / 3 | 732,381 | 1402.9 / 103.5 | 1449.1 / 180.4 | 1460.0 / 220.7 | 1449.1 / 347.3 | full-window run |

Each C cell is `prefill / full-window aggregate decode` tok/s; prefix caching
was disabled.

## Qwen3.6-35B-A3B-FP8

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/w8a16/normal/nomtp-fp16kv-1x256k-text-only.env` | normal | 256K | FP16 | 0 | text-only | 273,586 | 7378 / 128.7 |
| `qwen35b/w8a16/normal/nomtp-fp16kv-1x136k-text-image.env` | normal | 136K | FP16 | 0 | text+image | 146,485 | 5965.8 / 127.6 |
