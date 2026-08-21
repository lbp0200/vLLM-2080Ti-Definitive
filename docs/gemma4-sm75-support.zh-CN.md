# Gemma4 SM75 支持说明

本文集中记录 Gemma4 路线细节，因此 `0.2.1-pre2` README 不再展开这些内容。
README 所说的 Gemma4 基础支持，仅表示当前 vLLM 树能够识别相关模型和运行时
路径；并不表示任意 Gemma4 checkpoint 已成为双 RTX 2080 Ti CUDA 13 的正式路线。

## 证据范围

`0.2.1-pre2` 基于 vLLM `v0.27.1`、CUDA 13.0 和 PyTorch 2.13。目前还没有任何
Gemma4 路线完成此分支所需的发布级 SM75 复验。下方实验结果来自旧的 `v0.1.x`
CUDA 12.8 / PyTorch 2.11 运行时，只能作为起点，不能视为 cu130 的兼容性、性能或
质量结论。

当前 SM75 提升目标是
[迁移验证报告](2080ti-0.2.1-pre-validation.md) 中的 Qwen3.8 27B 路线。Gemma4
必须完成同等的构建、CUDA Graph、输出质量与性能测试后，才能并入主测试集合。

## 模型形态

上游 vLLM 模型注册表包含 Gemma4 的语言模型和多模态模型家族。在本 fork 中，
应以 checkpoint 的 `model_type` 与 processor metadata 为准，不能把 Gemma4
checkpoint 强行作为通用 draft model 加载。

| 模型形态 | 预期用途 | SM75 `0.2.1-pre2` 状态 |
| --- | --- | --- |
| `Gemma4ForCausalLM` | 纯文本 Gemma4 checkpoint | 基础运行时支持；无已提升的 SM75 preset |
| `Gemma4ForConditionalGeneration` | 带 tower 的文本/图像/视频/音频变体 | 基础模型支持；无已验证 SM75 多模态 preset |
| `Gemma4UnifiedForConditionalGeneration` | 无 encoder 的 unified 变体 | 基础模型支持；无已验证 SM75 多模态 preset |
| Gemma4 assistant model types | MTP/speculative assistant checkpoint | 必须走 Gemma4 MTP 路径；无 `0.2.1-pre2` SM75 提升证据 |

模型清单和多模态语义应以上游模型注册表与 processor metadata 为准；该清单不是
硬件专属验证表。

## 历史实验路线

以下路线明确为实验性路线。它们是本 fork 保留的 Gemma4 checkpoint 记录，但不
会出现在 `0.2.1-pre2` 的已测试权重表中。

| Target checkpoint | 权重量化 | 历史路线状态 | `0.2.1-pre2` 解读 |
| --- | --- | --- | --- |
| [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) 与 [google/gemma-4-31B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) | QAT target + 对应 assistant | 历史上用于 FP16/default-KV 探索和 assistant MTP 的首选实验目标 | 先复验 target-only 启动与输出，再复验配对 MTP 路线 |
| [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | GPTQ-INT4 | 历史实验路线 | 使用前需复验量化加载、graph capture、输出质量和 KV 容量 |

## MTP 与 Assistant 限制

Gemma4 assistant checkpoint 是模型专用的 MTP speculator，不是通用 draft
model。必须使用 vLLM 的 Gemma4 MTP 方法和 checkpoint 匹配的 assistant metadata，
主 target 也必须与该 assistant 兼容。target-only 启动成功，不能证明 MTP 已正确
配置。

在 SM75 上提升任意 assistant 路线前，必须在物理 TP=2 NVLink 双卡上验证：

- target-only 输出质量；
- target 加 assistant 的启动和生成；
- `enforce_eager=False` 下的 CUDA Graph capture；
- 冷态 4K/128 prefill 与 decode；以及
- 真实质量探针，而不是仅测试重复 token 的合成输出。

当前 MTP 实现和 assistant model 支持必须结合对应 vLLM 源码与验证报告确认后才能晋升。

## 历史 KV 与运行时说明

下表是 `v0.1.x` 实验的归档路线矩阵，仅用于避免把旧观察误读为当前 cu130 保证。

| 项目 | FP16/default KV | INT8 KV | TurboQuant KV |
| --- | --- | --- | --- |
| Marlin 权重实验 | 观察到 GPTQ 和 QAT 路线 | 部分/实验性 | 部分/实验性 |
| MTP | 探索过 QAT assistant MTP3 | 无正式 preset | 无正式 preset |
| 容量观察 | 曾观察到约 170K KV 余量 | 曾出现初始化问题 | 曾出现容量不足 |
| CUDA Graph | 实验路线中观察到 | 曾出现 fallback 问题 | 曾受到 admission/空间限制 |
| 快速 prefill | 探索过 FlashInfer 路径 | 实验性 | 实验性 |
| 多模态服务 | 无已验证 SM75 preset | 无已验证 SM75 preset | 无已验证 SM75 preset |

这些观察不能直接作为 profile 默认值。内核选择、显存预算、model runner 行为和
speculative decoding 在旧 cu128 分支与本 cu130 迁移之间均已变化。

## 多模态范围

上游能够识别 Gemma4 多模态模型形态，包括图像和其他 processor 处理的输入路径。
本 fork 尚未验证 SM75 `0.2.1-pre2` 的 text+image、视频或音频部署 preset。多模态
请求的 encoder 行为也必须与纯文本 decoder CUDA Graph 结果分开测试。

## 提升条件

任意 Gemma4 路线只有在双 RTX 2080 Ti 22 GB、TP=2 下具备以下全部证据后，才能
加入主已测试权重列表：

1. 使用 SM75 kernel 的全新 CUDA 13.0 源码构建。
2. 非 eager 的 target-only 生成，并通过质量探针。
3. CUDA Graph capture 与重复请求稳定性。
4. 记录精确 KV dtype 和 MTP 配置的可复现冷态 4K/128 prefill/decode 数据。
5. 对多模态或 MTP，分别提供 encoder 或 assistant 路径证据。

在此之前，Gemma4 只能称为基础支持或实验路线，不能称为已提升的 SM75 部署路线。
