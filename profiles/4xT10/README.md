# 4xT10 Profiles

Hardware-specific profiles for four 16 GiB Tesla T10 GPUs over PCIe (TP=4).
These routes require the ABI-matched PCIe custom all-reduce extension described
in the validation documents.

The layout is `qwen27b/w8a16/{fast,normal}`. All current routes are single
concurrency, 256K context, and multimodal:

| Profile | Mode | KV | Decoder | Messages |
|---|---|---|---|---|
| `qwen27b/w8a16/fast/mtp-fp16kv-1x256k-text-image.env` | fast | FP16 | MTP3 | text+image |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-image.env` | fast | TQK8V4 | MTP3 | text+image |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x256k-text-image.env` | normal | FP16 | MTP3 | text+image |
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x256k-text-image.env` | normal | FP16 | no MTP | text+image |

## Measured Performance

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | 0 | text+image | 351,319 | 1065.22 / 51.65 |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | 3 | text+image | 312,585 | 1440.31 / 73.68 |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-image.env` | fast | 256K | TQK8V4 | 3 | text+image | 780,814 | 1692.12 / 104.78 |

The measurements use four 16 GiB Tesla T10 GPUs over PCIe (TP=4), with the
ABI-matched PCIe CAR extension. They passed startup and an image request; the
reported performance is the median of three independent 4K/128 requests.
