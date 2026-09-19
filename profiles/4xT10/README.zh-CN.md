# 4xT10 Profile

## Qwen3.8-27B-FP8

测试权重：[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)；DFlash2 draft：[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x256k-text-only.env` | 256K | FP16 | DFlash2/7 | text-only | 300,304 | 1433.91 / 191.89 | 1536.13 / 189.38 |
| `qwen27b/w8a16/dflash2-fp16kv-1x235k-text-image.env` | 235K | FP16 | DFlash2/7 | text+image | 241,215 | 1443.96 / 190.89 | 1532.96 / 187.85 |
| `qwen27b/w8a16/mtp4-fp16kv-1x256k-text-image.env` | 256K | FP16 | MTP/4 | text+image | 320,484 | 1484.30 / 106.04 | 1527.71 / 104.25 |
| `qwen27b/w8a16/mtp3-fp8kv-2x256k-text-image.env` | 2 x 256K | FP8 | MTP/3 | text+image | 621,102 | 1466.79 / 96.05 | 1496.23 / 93.72 |

## 说明

1. Profile 直接放在各模型/权重目录下，不再按 mode 分目录。Mode 由 launcher 选择，默认使用 `fast`。
2. 性能数据统一使用 launcher 的可复测参考口径：仅在测试期间关闭 Prefix Cache、单次只发送一个纯文本请求、预热不计入统计、4K/128 取三次中位数，并完整运行 32K/512。图文 Profile 同样使用纯文本性能口径，图像语义另行验证。`4K/128` 表示准确的 4,096 输入 token，`32K/512` 表示准确的 32,768 输入 token；原始日志和请求 JSON 保存在仓库外部的内部审计目录。
3. 测试环境：2026-09-19，软件版本 v0.2.1。测试主机为双路 Intel Xeon E5-2673 v4（共 80 个逻辑 CPU），内存 60 GiB，Swap 8 GiB。测试拓扑使用物理 GPU 0、2、3、4，四张 Tesla T10（每张 16,384 MiB）；四卡位于同一 NUMA 节点，卡间为 PCIe PIX 连接，服务使用 TP4。NVIDIA 驱动版本为 595.91.07。运行环境为仓库的 `vllm-sm75-tp2-cu130`（CUDA 13.0，torch 2.13.0+cu130）。
