# 4xT10 Profile

四张 16 GiB Tesla T10（PCIe、TP=4）专用 profile。相关路线依赖验证文档中说明的
ABI 匹配 PCIe custom all-reduce 扩展。

当前目录为 `qwen27b/w8a16/{fast,normal}`。现有路线均为单并发、256K 上下文和多模态：

| Profile | 模式 | KV | 解码 | 消息 |
|---|---|---|---|---|
| `qwen27b/w8a16/fast/mtp-fp16kv-1x256k-text-image.env` | fast | FP16 | MTP3 | text+image |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-image.env` | fast | TQK8V4 | MTP3 | text+image |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x256k-text-image.env` | normal | FP16 | MTP3 | text+image |
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x256k-text-image.env` | normal | FP16 | no MTP | text+image |

性能和质量证据见[历史验证记录](../../docs/2080ti-0.2.1-pre-validation.md)。
