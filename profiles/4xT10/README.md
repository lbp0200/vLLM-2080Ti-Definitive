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

See the [historical validation record](../../docs/2080ti-0.2.1-pre-validation.md)
for performance and quality evidence.
