# 2x2080Ti Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | Mode | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x144k-text-only.env` | normal | 144K | FP16 | None (autoregressive) | text-only | 153,600 | 1700.42 / 33.92 | 1487.75 / 31.44 |
| `qwen27b/w8a16/fast/mtp5-fp8kv-1x256k-text-only.env` | fast | 256K | FP8 | MTP/5 | text-only | 307,041 | 1653.09 / 97.35 | 1406.51 / 101.96 |
| `qwen27b/w8a16/normal/mtp-fp8kv-1x220k-text-image.env` | normal | 220K | FP8 | MTP/3 | text+image | 228,224 | 1651.89 / 78.62 | 1405.63 / 79.93 |
| `qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env` | normal | 256K | FP8 | MTP/3 | text-only | 320,232 | 1653.31 / 80.17 | 1409.44 / 81.52 |

## Qwen3.8-27B-NVFP4

Tested weight: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Mode | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen27b/w4a16/fast/dflash2-fp16kv-1x256k-text-only.env` | fast | 256K x1 | FP16 | DFlash2 (default K=7) | text-only | 376,832 | 1465.02 / 220.69 | 1280.57 / 213.66 |
| `qwen27b/w4a16/fast/dflash2-tqk8v4-2x172k-text-only.env` | fast | 172K x2 | TQK8V4 | DFlash2 (default K=7) | text-only | 384,474 | 1499.70 / 145.40 | 1314.00 / 117.00 |
| `qwen27b/w4a16/normal/mtp-fp8kv-1x240k-text-image.env` | normal | 240K | FP8 | MTP/3 | text+image | 410,093 | 1442.55 / 73.96 | 1258.70 / 75.56 |

## Qwen3.6-35B-A3B-FP8

Tested weight: [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | Mode | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen35b/w8a16/normal/nomtp-fp16kv-1x256k-text-only.env` | normal | 256K | FP16 | None (autoregressive) | text-only | 266,305 | 6811.75 / 120.15 | 5981.31 / 110.54 |

`4K/128` means about 4K input tokens and 128 output tokens, while `32K/512` means about 32K input tokens and 512 output tokens; both use synthetic inputs designed for high speculative acceptance and are for like-for-like throughput comparison, not representative of real-world inference. Unless noted above, 4K/128 is the median of three post-warmup requests and 32K/512 is one post-warmup request; start the selected profile with `launcher.sh` to reproduce them.
