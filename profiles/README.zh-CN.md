# Profile 导引

语言：[English](README.md) | 简体中文

这里是 vLLM 2080Ti Definitive 自带的启动 profile。一个 profile 只是运行参数
的 `.env` 预设，不包含模型权重路径；权重目录通过 `launcher.sh` 或
`MODEL_DIR=...` 单独选择。

下面这些 profile 是从维护中的 v0.1.x CUDA 12.8 / Torch 2.11 路线保留的
launcher 兼容模板。表中的历史吞吐数字不是 0.2.1-pre3 的 promotion 证据；
当前 cu130 验证结果见 `docs/2080ti-0.2.1-pre-validation.md`。

本说明描述的是 profile 兼容性，而不是 `0.2.1-pre3` 的已测试 checkpoint 矩阵。
当前刻意收窄后的清单位于仓库根目录 `README.md`：Qwen3.8 27B 候选 checkpoint
以及保留的 Qwen3.x 35B FP8 记录。Gemma4 路线细节单列在
`docs/gemma4-sm75-support.zh-CN.md`。

这里列出的上下文容量和吞吐数据，验证硬件是双 RTX 2080 Ti 22GB，tensor
parallel size 2。

目录结构：

```text
profiles/
  templates/
  qwen27b/
    normal/
      fp8/
      int4/
    fast/
      fp8/
      int4/
    experimental/
      int8-w8a8/
    user/
  qwen3.8-27b/
    normal/
      fp8/
      nvfp4/
    fast/
      fp8/
      nvfp4/
  qwen35b/
    normal/
      fp8/
    aggressive/
      fp8/
    fast/
      fp8/
    user/
```

启动模式：

- `safe`：保守回退模式，优先保证可用性。
- `normal`：推荐日常模式，适合稳定部署。
- `fast`：高性能模式，适合追求更高吞吐的场景。
- `aggressive`：更加激进的模式，性能与质量风险最高。

`profiles/templates/` 存放可选 chat template 预设。它们通过 launcher 作为全局
服务设置选择；具体 route profile 不保存 chat template、GPU、端口、reasoning
默认值或工具调用默认值。
对内置 Qwen3/Qwen3.x 路线，launcher 会在未显式设置时补上 `qwen3`
reasoning parser，让启动 smoke 和聊天解析都跟模型默认 reasoning 行为保持一致。
如需诊断无 reasoning parser 路径，可设置 `REASONING_PARSER=off`。

文件名描述路线：

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV 精度定位：

- `fp16kv`：质量路线。
- `int8kv`：容量 / 平衡路线；当前只作为 `normal` profile 保留。
- `tqk8v4`：TurboQuant K8V4 压缩路线；当前只保留质量通过的 `fast` profile。
- 官方 Qwen3.x 35B 当前提供 FP8 权重 + FP16 KV 的纯文本和图文预设。

内置 TQK8V4 profile 使用 `MAX_BATCHED_TOKENS=2560`，这是 Qwen hybrid cache
block 对齐后的 prefix-cache 路径已验证设置。

## 已验证 Profile

### Qwen3.x 27B FP8（旧容量基线）

这些 profile 容量与历史数字早于 `0.2.1-pre2`；为 cu130 选择 checkpoint 前，请先
查看根目录的已测试权重清单。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 1 | 1529.33 / 30.42 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |

### Qwen3.8 27B 多模态 MTP3（0.2.1-pre2 正式 profile）

以下路线在 NVLink 连接的双 RTX 2080 Ti、TP=2 上验证。每条路线均先通过自然图像问答：
正确识别蓝色方块、橙色圆和 `K7P`。保留的图文路线吞吐为三轮请求中的最高 prefill / decode
值；每轮为 4K 输入、128 输出，并使用不同 prompt 前缀避开 prefix cache。性能请求仅为固定
128 token decode 窗口设置 `ignore_eos`。
FP8 normal 行在提升至 104K 时使用固定输出的合成 4K/128 请求复测；图像问答质量门槛另在下文
单独记录。

| Checkpoint | Profile | 上下文 | KV | 吞吐性能 |
|---|---|---:|---|---:|
| Qwen3.8-27B-FP8 | `qwen3.8-27b/normal/fp8/fp16kv-104K-mtp3-text-image.env` | 104K | FP16 | 1506.86 / 82.88 |
| Qwen3.8-27B-FP8 | `qwen3.8-27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | 240K | TQK8V4 | 1355.04 / 73.48 |
| Qwen3.8-27B-NVFP4 | `qwen3.8-27b/normal/nvfp4/fp8kv-240K-mtp3-text-image.env` | 240K | FP8 | 1266.09 / 55.69 |
| Qwen3.8-27B-NVFP4 | `qwen3.8-27b/fast/nvfp4/tqk8v4-240K-mtp3-text-image.env` | 240K | TQK8V4 | 1276.92 / 83.99 |

FP8 normal 路线使用 `GPU_UTIL=0.96`，启动时分配 4.04 GiB，并在 CUDA Graph
profiling 后实测 GPU KV 为 110,784 tokens；留出 block 余量后将正式上下文设为
104K。三次 4K/128 合成请求均返回 128/128，最高实测 prefill / decode 为
1506.86 / 82.88 tok/s。

normal NVFP4 使用 `compressed-tensors`，在 SM75 上由 Marlin 处理 NVFP4 与权重仅 FP8
线性层，并显式使用 FP8 KV。其 240K 受 checkpoint 的 262,144 token 位置上限约束，而不是
KV 显存：实测 GPU KV 为 426,080 tokens。fast NVFP4 实测 GPU KV 为 515,723 tokens，但同样
受该位置上限约束。两条 fast 路线均使用 CUDA Graph decode；NVFP4 fast 同时捕获 FULL decode
和 PIECEWISE 混合 prefill/decode 图。

### Qwen3.8 27B FP8 纯文本（0.2.1-pre2 正式 profile）

以下 profile 用于官方 `Qwen/Qwen3.8-27B-FP8`，启用 `LANGUAGE_MODEL_ONLY=1` 与
`SKIP_MM_PROFILING=1`。它们与上方图文 profile 独立，并保留 CUDA Graph 执行。

| Profile | 上下文 | KV | MTP | CUDA Graph | 实测 GPU KV tokens | 4K/128 prefill / decode tok/s |
|---|---:|---|---:|---|---:|---:|
| `qwen3.8-27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | 128K | FP16 | 3 | PIECEWISE，size 4 | 138,394 | 1496.95 / 83.90 |
| `qwen3.8-27b/normal/fp8/fp16kv-144K-nomtp-text-only.env` | 144K | FP16 | 0 | FULL_AND_PIECEWISE，size 1 | 152,749 | 1501.39 / 30.40 |
| `qwen3.8-27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | 256K | TQK8V4 | 3 | FULL_AND_PIECEWISE，size 4 | 310,827 | 1525.37 / 83.51 |

每条路线均通过 `PROFILE_OK` 探针和三组不同的 4K/128 合成测试，且完整输出
128 token；吞吐取其中最高有效值。

### Qwen3.8 27B NVFP4 纯文本（0.2.1-pre2 正式 profile）

以下路线使用 `Qwen3.8-27B-NVFP4`、TP=2 和 `LANGUAGE_MODEL_ONLY=1`。normal
模式明确使用 FP8 KV，fast 模式使用 TurboQuant K8V4。两条路线均通过
`PROFILE_OK` 质量探针，并在固定 token 的 4K/128 合成测试中严格生成
128/128；吞吐取三组不同 prompt 变体中的最高有效值。

| Profile | 上下文 | KV | MTP | CUDA Graph | 实测 GPU KV tokens | 4K/128 prefill / decode tok/s |
|---|---:|---|---:|---|---:|---:|
| `qwen3.8-27b/normal/nvfp4/fp8kv-240K-nomtp-text-only.env` | 240K | FP8 | 0 | PIECEWISE + FULL decode | 530,720 | 1421.10 / 38.89 |
| `qwen3.8-27b/fast/nvfp4/tqk8v4-240K-mtp3-text-only.env` | 240K | TQK8V4 | 3 | PIECEWISE + FULL decode | 564,130 | 1411.91 / 102.60 |

### Qwen3.8 27B NVFP4 真并发（0.2.1-pre3 正式 profile）

`qwen3.8-27b/normal/nvfp4/fp8kv-16K-nomtp-concurrent.env` 是可选吞吐路线。
它使用 FP8 KV、无 MTP、`MAX_BATCHED_TOKENS=8192`、`MAX_NUM_SEQS=8`、
512-token long-prefill threshold、shared prefill frontier，并关闭 prefix cache。
严格 4K/128 下，并发 1/2/4/8 的完整窗口 aggregate decode 是
`41.53 / 79.94 / 151.19 / 269.30 tok/s`。它不是容量 profile，不能替代上面的
240K 时延导向路线。

### Qwen3.x 35B FP8（旧容量基线）

保留的 Qwen3.x 35B FP8 checkpoint 记录见根目录 README；提升 profile 前仍须单独
完成 cu130 复验。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 6705.13 / 97.33 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 1 | 5485.13 / 95.20 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | aggressive | 256K | FP16 | 0 | text-only | 1 | 6843.01 / 124.01 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | aggressive | 136K | FP16 | 0 | text+image | 1 | 5422.83 / 124.11 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | fast | 178K | FP16 | 3 | text-only | 1 | 5889.20 / 195.95 |

### Qwen3.x 27B INT4（旧容量基线）

这些保留的 INT4 profile 数字不构成 `0.2.1-pre2` 的已测试 checkpoint 说明。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 等待 0.2.1-pre2 重测 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 每工作区 250K | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 每工作区 250K | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

### Qwen3.8 27B INT8 W8A8（实验性部署路线）

以下 profile 按双 2080 Ti 的实测显存容量设置为 32K，并为 MTP3 图和激活保留余量，
对应 `RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP`。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 |
|---|---|---:|---|---:|---|---:|
| `qwen27b/experimental/int8-w8a8/fp16kv-32K-mtp3-text-only.env` | normal | 32K | FP16 | 3 | text-only | 1 |
| `qwen27b/experimental/int8-w8a8/fp16kv-32K-nomtp-text-only.env` | normal | 32K | FP16 | 0 | text-only | 1 |
| `qwen27b/experimental/int8-w8a8/tqk8v4-32K-mtp3-text-only.env` | fast | 32K | TQK8V4 | 3 | text-only | 1 |

### Qwen3.8 27B NVFP4（实验性 ModelOpt 路线）

这些 profile 用于 `pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB`，必须配套其 ModelOpt
`MIXED_PRECISION` 权重文件。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 |
|---|---|---:|---|---:|---|---:|
| `qwen27b/experimental/nvfp4/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 |
| `qwen27b/experimental/nvfp4/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 |
| `qwen27b/experimental/nvfp4/tqk8v4-128K-mtp3-text-only.env` | fast | 128K | TQK8V4 | 3 | text-only | 1 |
