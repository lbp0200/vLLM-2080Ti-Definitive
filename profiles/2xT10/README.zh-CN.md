# 2xT10 Profile

## Qwen3.8-27B-INT4

测试权重：[RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/mtp3-fp8kv-1x262K-text-only.env` | 262K | FP8 | MTP/3 | text-only | 266,537 | 1122.00 / 87.88 | 1070.10 / 83.57 |
| `qwen27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env` | 2 x 155K | TQK8V4 | MTP/3 | text-only | 321,789 | 1138.22 / 87.42 | 1000.18 / 59.55 |
| `qwen27b/w4a16/mtp3-tq4nc-2x220K-text-only.env` | 2 x 220K | TQ4NC | MTP/3 | text-only | 471,807 | 1134.92 / 83.54 | 1100.51 / 49.76 |
| `qwen27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env` | 196K | TQK8V4 | MTP/3 | text+image | 206,888 | 1138.74 / 87.46 | 1053.94 / 50.20 |

## 说明

1. Profile 直接平铺在模型/权重目录下；启动模式由 launcher 选择，默认使用 `fast`。
2. 性能数据统一使用 launcher 的可复测参考口径：仅在测试期间关闭 Prefix Cache、单次只发送一个纯文本请求、预热不计入统计、4K/128 取三次中位数，并完整运行 32K/512。图文 Profile 同样使用纯文本性能口径，图像语义另行验证。`4K/128` 表示准确的 4,096 输入 token 和 128 输出 token；`32K/512` 表示准确的 32,768 输入 token 和 512 输出 token。
3. 测试环境：2026-09-25 使用 v0.2.2 runtime development build 验证。测试主机为双路 Intel Xeon E5-2673 v4（共 80 个逻辑 CPU），内存 60 GiB，Swap 8 GiB。测试拓扑使用物理 GPU 0、2，两张 Tesla T10（每张 16,384 MiB）；两卡位于同一 NUMA 节点，卡间为 PCIe PIX 连接，服务使用 TP2。NVIDIA 驱动版本为 595.91.07。运行环境使用 CUDA 13.0 和 torch 2.13.0+cu130。
4. 多模态路线启用视觉分支后 196K 启动成功，并正确描述了 JPEG 流程图中的 proxy/prefill/decode 处理链路。
