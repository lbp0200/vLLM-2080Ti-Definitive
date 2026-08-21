<!-- markdownlint-disable MD001 MD041 -->
# vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-cover.jpg)

面向双 RTX 2080 Ti 22 GB / SM75 推理的硬件定向 vLLM fork。本分支是 `vllm-2080ti-definitive-0.2.x` 维护线：基于上游 vLLM `v0.27.1`、CUDA 13.0 和 PyTorch 2.13 的公开 `0.2.x` 预发布线。它不是稳定生产默认版本；持续维护的 `0.1.x` 线仍是项目当前的稳定发布主线，对应 CUDA 12.8 与 PyTorch 2.11。

项目保留复现双 2080 Ti TP=2 栈所需的 SM75 专用源码修改、launcher profile 和
测试证据。它基于上游 vLLM；再发布派生版本时必须保留上游许可证、上游署名以及
`github.com/weicj` 的项目署名。

语言：[English](README.md) | 简体中文

![单请求实时测速演示](docs/assets/vllmspeed.gif)

Fork 版本：`0.2.1-pre2`
基础 vLLM：`0.27.1`

分支：[`vllm-2080ti-definitive-0.2.x`](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/vllm-2080ti-definitive-0.2.x)
预发布快照：[v0.2.1-pre2](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.1-pre2)
相比 pre1 的变化：[CHANGELOG.md](CHANGELOG.md)

## 为什么用 RTX 2080 Ti 做 LLM 推理？

这个项目的判断很实际：两张通过 NVLink 连接的 22 GB RTX 2080 Ti 提供 44 GB
显存、较高的显存带宽和 136 个 Turing SM。经过针对 SM75 的 vLLM 适配后，这套
硬件不只是运行小模型，也足以承载严肃的本地 27B 与 35B 级模型服务。

| 指标 | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB 基线 | 倍率 |
| --- | ---: | ---: | ---: |
| 物理 CUDA core | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| 物理 Tensor Core | 1,088 | 336 | 3.24x |
| Dense FP16 矩阵吞吐 | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存 | 44 GB | 24 GB | 1.83x |

本 fork 通过 Marlin、FlashInfer/FlashQLA、TurboQuant/INT8 KV、MTP 和 CUDA
Graph，把这些硬件资源转成可用的 serving 栈。

## 当前状态

`0.2.x` 的目标环境是 Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15、CUDA 13.0 与 PyTorch 2.13。持续维护的 `0.1.x` 线仍是 CUDA 12.8、PyTorch 2.11、较早 kernel 以及 GCC 12/13/14 的兼容路线。

双 2080 Ti 的 CUDA Graph 验证记录在
[迁移验证报告](docs/2080ti-0.2.1-pre-validation.md)：其中包含实际显卡选择规则、
构建门槛、正确性回归以及 4K/128 测试口径。仅能加载的 checkpoint 不应被视为
已提升为部署路线。

此前已合并 SM75 PR 的迁移判断见 [0.2.x PR 迁移审计](docs/0.2.x-pr-migration-audit.md)。

## 核心路线

服务形态：

- 目标是双 2080 Ti 上的极限单并发：一个个人 agent 场景、一个严肃的 27B 或
  35B 模型，以及这套硬件能稳定承载的最大实用上下文。
- 这不是多租户 serving 集群。长 prefill 在调好参数后可以安全承载，但在 TP=2
  runtime scheduler 中实际会被串行化。

状态说明：已验证表示有对应证据；实验表示只有部分或历史证据；不支持表示已知
存在缺失路径。

### Qwen3.8 27B

Qwen3.8 27B 是当前 SM75 验证主线。下表每一行均通过列出的项目内 profile，在物理双
2080 Ti NVLink、TP=2、非 eager CUDA Graph 下启动。数据取三次独立 4K 输入、128 输出、
不同 prompt 请求中的最高有效 prefill / decode；prefix cache 命中和质量探针失败的运行
不计入。

| Checkpoint | 已发布 profile（`PROFILE`、`MODE`） | 4K/128 prefill / decode tok/s |
| --- | --- | ---: |
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `qwen3.8-27b/normal/fp8/fp16kv-128K-mtp3-text-only.env`、`normal` | 1496.95 / 83.90 |
| Qwen/Qwen3.8-27B-FP8 | `qwen3.8-27b/normal/fp8/fp16kv-144K-nomtp-text-only.env`、`normal` | 1501.39 / 30.40 |
| Qwen/Qwen3.8-27B-FP8 | `qwen3.8-27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env`、`fast` | 1525.37 / 83.51 |
| Qwen/Qwen3.8-27B-FP8 | `qwen3.8-27b/normal/fp8/fp16kv-104K-mtp3-text-image.env`、`normal` | 1506.86 / 82.88 |
| Qwen/Qwen3.8-27B-FP8 | `qwen3.8-27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env`、`fast` | 1355.04 / 73.48 |
| [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | `qwen3.8-27b/normal/nvfp4/fp8kv-240K-nomtp-text-only.env`、`normal` | 1421.10 / 38.89 |
| unsloth/Qwen3.8-27B-NVFP4 | `qwen3.8-27b/fast/nvfp4/tqk8v4-240K-mtp3-text-only.env`、`fast` | 1411.91 / 102.60 |
| unsloth/Qwen3.8-27B-NVFP4 | `qwen3.8-27b/normal/nvfp4/fp8kv-240K-mtp3-text-image.env`、`normal` | 1266.09 / 55.69 |
| unsloth/Qwen3.8-27B-NVFP4 | `qwen3.8-27b/fast/nvfp4/tqk8v4-240K-mtp3-text-image.env`、`fast` | 1276.92 / 83.99 |

FP8 和 Unsloth NVFP4 纯文本路线都通过 `PROFILE_OK` 质量探针，且实测请求严格输出
128/128。官方 FP8 路线使用 Marlin weight-only FP8 与 FP16 KV 或 TurboQuant K8V4；
Unsloth checkpoint 使用 `compressed-tensors`、SM75 Marlin 的 NVFP4/FP8 线性层子集，
并按表格使用 FP8 KV 或 TurboQuant K8V4。

以下候选仅有短路线证据，属于**实验路线**，不是已提升的部署 profile：
`pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB` 是不同的 ModelOpt 格式，使用其专属
`qwen27b/experimental/nvfp4` profile；`RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP`
使用 32K 的 `qwen27b/experimental/int8-w8a8` profile。它们不能继承正式 FP8/Unsloth
路线的质量或容量结论。MXFP4 在 SM75 上仍不支持，因为上游未为该架构生成所需 Marlin
kernel 变体。

完整的 profile-first 证据矩阵见[测试征集](docs/0.2.1-pre-testing-call.zh-CN.md)，
实际 graph mode、受控对比、回归测试和已知限制见[迁移验证报告](docs/2080ti-0.2.1-pre-validation.md)。

### Qwen3.x 35B FP8

35B FP8 路线保留自已验证的 `v0.1.x` 双 2080 Ti profile 集。它们仍可作为兼容
参考，但在提升为 `0.2.1-pre2` 部署预设前必须独立完成 cu130 验证。

保留的预设覆盖 FP16 KV 256K 纯文本 `normal` / `aggressive`、FP16 KV 136K
图文 `normal` / `aggressive`，以及一条 178K `fast` MTP3 profile。

### Gemma4 基础支持

该树具备 Gemma4 的基础模型和运行时支持，但 Gemma4 不在当前 `0.2.1-pre2` 的
SM75 发布验证集合中。checkpoint 类型、MTP 要求、KV cache 限制、多模态状态，
以及历史证据与 cu130 证据的区别，均见
[Gemma4 SM75 支持说明](docs/gemma4-sm75-support.zh-CN.md)。

## 已测试模型权重

这是刻意收窄后的 `0.2.x` checkpoint 清单。"已验证"表示列出的项目内 profile 已在物理
双 2080 Ti NVLink 上完成；不代表该 checkpoint 的所有上下文、KV dtype 或 MTP 配置
都已验证。35B 一行保留的是 `v0.1.x` 证据，明确不构成 cu130 的提升依据。

| 模型路线 | 权重路线 | 模型卡 | 状态 |
| --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 已验证正式纯文本和图文 profile |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 已验证正式纯文本和图文 profile |
| Qwen3.8 27B | NVFP4 | [pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB](https://huggingface.co/pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB) | 仅实验性 ModelOpt 短路线证据 |
| Qwen3.8 27B | INT8 W8A8 | [RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) | 仅实验性 32K 短路线证据 |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)<br>[Jackrong/Qwopus3.6-35B-A3B-Coder-FP8](https://huggingface.co/Jackrong/Qwopus3.6-35B-A3B-Coder-FP8)<br>[kyr0/Ornith-35B-FP8-E4M3-MTP](https://huggingface.co/kyr0/Ornith-35B-FP8-E4M3-MTP) | `v0.1.x` 已验证；cu130 待复验 |

## 构建与启动

下载仓库后使用迁移构建入口：

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive
git switch --track origin/vllm-2080ti-definitive-0.2.x
./build.sh
```

`build.sh` 会创建 `.venv`、安装目标依赖、编译 CUDA 扩展，并把构建输出记录到
`build-logs/`。目标编译器、kernel 或 CUDA 条件不满足时，它会明确失败。默认编译
并发会根据主机 CPU 与内存自动选择；仅在确有需要时通过 `MAX_JOBS` 或
`BUILD_MAX_JOBS` 显式覆写。

构建成功后通过 launcher 启动并管理服务：

```bash
./launcher.sh
```

交互式 launcher 可以选择 checkpoint、profile、模式、GPU/TP、端口、服务范围、
chat template、reasoning 默认值和工具调用设置，并显示服务状态、PID、API URL、
日志路径、prefix-cache 状态和已上报的 cache 容量。checkpoint 路径与 profile 分开
选择；混合 GPU 主机上应先固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，再选择物理
2080 Ti 的 GPU ID。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
MODEL_DIR=/path/to/checkpoint \
PROFILE=qwen3.8-27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env \
MODE=fast GPU_DEVICES=4,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

profile 只保存路线参数。容量与历史吞吐记录见
[profiles/README.zh-CN.md](profiles/README.zh-CN.md)。修改路线后先运行
`launcher.sh --print-config` 检查实际配置。

## Profile 与推荐路线

从 [Profile 导引](profiles/README.zh-CN.md) 开始选。Profile 按
`profiles/<model>/<mode>/<weight>/<route>.env` 组织，例如
`qwen3.8-27b/normal/fp8/fp16kv-128K-mtp3-text-only.env`、
`qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` 和
`qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env`。

可用模式：

- `normal`：稳定的日常部署模式。
- `fast`：更高性能模式，只用于已验证路线。
- `aggressive`：性能最高但质量风险也最高。
- `safe`：用于排障和兼容性的保守回退模式。

Profile 只选择路线参数。GPU、端口、模型路径、chat template 和 reasoning 默认值
由 launcher 统一管理。

## MTP 与 KV 精度

优先使用项目自带 profile，不要一开始手动调 MTP 和 KV。KV 先按目标选择：
FP16/default KV 追求输出质量，INT8 KV 用于平衡型长上下文服务，TurboQuant K8V4
用于压缩 fast 路线。MTP 收益取决于接受率，合成峰值必须再用真实输出和质量探针
检查。

当前迁移请以[验证报告](docs/2080ti-0.2.1-pre-validation.md)中的精确方法和数据
为准，尤其是 TurboQuant 与 MTP3。历史 profile 容量不是 cu130 证据。

## 目标硬件

- 两张经 NVLink 连接的 RTX 2080 Ti 22 GB
- NVIDIA Turing / SM75，tensor parallel size 2
- `0.2.1-pre2` 目标：CUDA 13.0、PyTorch 2.13、Python 3.12
- 目标主机：Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15

其它 Turing 显卡仍需针对显存容量、PCIe/NVLink 拓扑、模型 head dimension、
KV cache dtype 和 CUDA Graph 行为独立验证。

## 硬件 Q&A

**需要什么样的卡间互联？**

推荐 NVLink。PCIe P2P 是底线，但没有 NVLink 时不能把窄 PCIe 链路直接视为已验证
替代方案；应先确认 P2P，再按实际主机拓扑测试。

**需要很强的 CPU 或很多内存吗？**

不需要高端 CPU，但现代单核性能和较低的平台延迟很重要。更多内存主要帮助构建、
下载和 compile cache；即使 GPU 相同，老旧 CPU 平台也可能降低 decode 吞吐。

**可以混用 11 GB 和 22 GB Turing 卡吗？**

不建议用于文档中的 27B/35B TP=2 路线。TP 会受到较小 rank 显存的限制。更好的
候选是成对的高显存 TU102 卡，例如 TITAN RTX、Quadro RTX 6000 或 Quadro RTX
8000，并且要有 NVLink 或确认可用的 PCIe P2P，之后仍需独立验证 profile。

**应该使用哪些 CUDA 和 PyTorch 版本？**

`0.2.1-pre2` 目标是 CUDA 13.0 + PyTorch 2.13。旧的 CUDA 12.8 + PyTorch 2.11
仍作为独立的 `v0.1.x` 兼容路线维护。PyTorch CUDA 构建、toolkit、FlashInfer/
FlashQLA 构建和启动 profile 必须保持一致，不能混用运行时假设。

**还有哪些硬件风险？**

注意散热、供电稳定性，以及模型和编译缓存所需的 SSD 空间。长 prefill 或反复
CUDA Graph/AOT 编译时降频很容易被误判为软件性能回退。

## 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双 2080 Ti
  模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。本仓库聚焦于补丁后
  的 vLLM runtime。

## 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向 fork，
遵循 Apache-2.0 license，保留上游项目结构，并加入面向双 2080 Ti 的 SM75 runtime
补丁、启动 profile 和验证记录。

当前使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving 框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：attention、sampling
  和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 Gated DeltaNet /
  Qwen hybrid linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  SM70/SM75 适配版本，用于已验证的 Qwen prefill 路线。
- TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM 相关 kernel。

上游更新合入后，仍会在本 fork 的 SM75 范围内重新验证。
