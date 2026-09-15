# 2x2080Ti Profile

两张 RTX 2080 Ti 专用 profile。当前模型族包括 Qwen 27B NVFP4
（`qwen27b/w4a16`）、Qwen 27B FP8（`qwen27b/w8a16`）和 Qwen 35B FP8
（`qwen35b/w8a16`）。

已验证的 DFlash2 路线：

| Profile | 模式 | KV | 并发/上下文 | 消息 |
|---|---|---|---|---|
| `qwen27b/w4a16/fast/dflash2-tqk8v4-2x172k-text-only.env` | fast | TQK8V4 | 2 x 172K | text-only |
| `qwen27b/w4a16/fast/dflash2-tqk8v4-1x256k-text-image.env` | fast | TQK8V4 | 1 x 256K | text+image |

其他路线仍按模型和权重目录保存。DFlash2 SM75 证据见[验证文档](../../docs/0.2.1-pre4-dflash2-sm75-validation.md)。
