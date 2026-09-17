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

## 完整测量数据

| Profile | 模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | 0 | text+image | 351,319 | 1065.22 / 51.65 |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | 3 | text+image | 312,585 | 1440.31 / 73.68 |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-image.env` | fast | 256K | TQK8V4 | 3 | text+image | 780,814 | 1692.12 / 104.78 |

数据来自四张 16 GiB Tesla T10（PCIe、TP=4），使用 ABI 匹配的 PCIe CAR 扩展。所有路线均通过启动和图像请求，性能为三个独立 4K/128 请求的中位数。
