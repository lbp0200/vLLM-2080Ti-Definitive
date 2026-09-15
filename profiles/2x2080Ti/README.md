# 2x2080Ti Profiles

Hardware-specific profiles for two RTX 2080 Ti GPUs. Available model families
are Qwen 27B NVFP4 (`qwen27b/w4a16`), Qwen 27B FP8 (`qwen27b/w8a16`), and Qwen
35B FP8 (`qwen35b/w8a16`).

Validated DFlash2 routes:

| Profile | Mode | KV | Concurrency/context | Messages |
|---|---|---|---|---|
| `qwen27b/w4a16/fast/dflash2-tqk8v4-2x172k-text-only.env` | fast | TQK8V4 | 2 x 172K | text-only |
| `qwen27b/w4a16/fast/dflash2-tqk8v4-1x256k-text-image.env` | fast | TQK8V4 | 1 x 256K | text+image |

Other measured routes remain under their model/weight directories. The DFlash2
SM75 evidence is in [the validation document](../../docs/0.2.1-pre4-dflash2-sm75-validation.md).
