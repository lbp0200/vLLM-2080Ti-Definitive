# 4xT10 Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | Mode | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen27b/w8a16/fast/dflash2-fp16kv-1x256k-text-only.env` | fast | 256K | FP16 | DFlash2/7 | text-only | 299,474 | 1444.73 / 190.78 | 1456.77 / 188.94 |
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | None (autoregressive) | text+image | 356,764 | 1518.31 / 40.22 | 1563.00 / 37.70 |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | MTP/3 | text+image | 315,343 | 1488.27 / 76.03 | 1514.79 / 77.53 |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-image.env` | fast | 256K | TQK8V4 | MTP/3 | text+image | 797,912 | 1444.72 / 113.78 | 1497.69 / 85.95 |

## Qwen3.8-27B-NVFP4

Tested weight: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

| Profile | Mode | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen27b/w4a16/fast/dflash2-fp16kv-2x256k-text-only.env` | fast | 2 x 256K | FP16 | DFlash2/7 | text-only | 696,320 | 1703.67 / 442.12 (C2 aggregate) | 1569.86 / 503.78 (C2 aggregate) |

The NVFP4 C2 row uses synchronized client submission. Prefill is aggregate prompt tokens divided by the last TTFT, and decode covers the full interval from the first token to the last completion. Matching C1 results were `1498.13 / 244.63` at 4K/128 and `1354.99 / 253.20` at 32K/512.

The measurements use four 16 GiB Tesla T10 GPUs over PCIe (TP=4) with the ABI-matched PCIe CAR extension. `4K/128` means about 4K input and 128 output tokens, while `32K/512` means about 32K input and 512 output tokens; both are high-speculative-acceptance synthetic tests. The 4K/128 result is the median of three post-warmup requests and 32K/512 is one post-warmup request; start the selected profile with `launcher.sh` to reproduce them.
