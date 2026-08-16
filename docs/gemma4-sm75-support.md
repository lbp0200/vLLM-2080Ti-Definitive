# Gemma4 SM75 Support Notes

This document contains the Gemma4 route detail deliberately kept out of the
`0.2.1-pre` README. The README's statement of baseline Gemma4 support means
that this vLLM tree recognizes the relevant model/runtime paths. It does not
mean that a Gemma4 checkpoint is a promoted dual-RTX-2080-Ti CUDA 13 route.

## Evidence Scope

`0.2.1-pre` is based on vLLM `v0.27.1`, CUDA 13.0, and PyTorch 2.13. No Gemma4
route has yet completed the release-level SM75 revalidation required for this
branch. The experimental results below were obtained on the previous `v0.1.x`
CUDA 12.8 / PyTorch 2.11 runtime. They are useful starting points, not cu130
compatibility, throughput, or quality claims.

The current SM75 promotion target is the Qwen3.8 27B lane documented in
[the migration report](2080ti-0.2.1-pre-validation.md). Keep Gemma4 separate
until it has equivalent build, CUDA Graph, output-quality, and benchmark
evidence.

## Model Forms

The upstream vLLM model registry includes the Gemma4 language and multimodal
model families. In this fork, use the checkpoint's `model_type` and processor
metadata as authoritative; do not force a Gemma4 checkpoint through a generic
draft-model route.

| Model form | Intended use | SM75 `0.2.1-pre` status |
| --- | --- | --- |
| `Gemma4ForCausalLM` | Text-only Gemma4 checkpoints | Baseline runtime support; no promoted SM75 preset |
| `Gemma4ForConditionalGeneration` | Tower-based text/image/video/audio variants | Baseline model support; no validated SM75 multimodal preset |
| `Gemma4UnifiedForConditionalGeneration` | Encoder-free unified variants | Baseline model support; no validated SM75 multimodal preset |
| Gemma4 assistant model types | MTP/speculative assistant checkpoints | Requires the Gemma4 MTP path; no `0.2.1-pre` SM75 promotion evidence |

For the upstream model inventory and multimodal semantics, see
[supported models](models/supported_models.md). That upstream inventory is not
a hardware-specific validation table.

## Historical Experimental Routes

These routes are intentionally marked experimental. They are the only Gemma4
checkpoint records retained by the fork, but are not included in the
`0.2.1-pre` tested-checkpoint list.

| Target checkpoint | Weight route | Historical route state | `0.2.1-pre` interpretation |
| --- | --- | --- | --- |
| [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) with [google/gemma-4-31B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) | QAT target plus matching assistant | Preferred historical experimental target for FP16/default-KV exploration and assistant MTP | Revalidate target-only startup/output first, then revalidate the paired MTP route |
| [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | GPTQ-INT4 | Historical experimental route | Revalidate quantization loading, graph capture, output quality, and KV capacity before use |

## MTP And Assistant Constraints

Gemma4 assistant checkpoints are model-specific MTP speculators. They are not
generic draft models. Use vLLM's Gemma4 MTP method and the checkpoint's matching
assistant metadata; the main target must be compatible with the assistant
checkpoint. A successful target-only startup is not evidence that MTP is
configured correctly.

Before promoting an assistant route on SM75, verify all of the following on the
physical TP=2 NVLink pair:

- target-only output quality;
- target plus assistant startup and generation;
- CUDA Graph capture with `enforce_eager=False`;
- cold 4K/128 prefill and decode measurements; and
- a real quality probe, not only repeated-token synthetic output.

The upstream MTP implementation details and supported assistant model types are
in [the Gemma4 MTP guide](features/speculative_decoding/mtp.md#gemma-4-assistant-models).

## Historical KV And Runtime Notes

The following is an archival route matrix from the `v0.1.x` experiments. It is
included to prevent the older observations from being mistaken for a current
cu130 guarantee.

| Area | FP16/default KV | INT8 KV | TurboQuant KV |
| --- | --- | --- | --- |
| Marlin weight experiments | GPTQ and QAT routes observed | Partial/experimental | Partial/experimental |
| MTP | QAT assistant MTP3 was explored | No shipped preset | No shipped preset |
| Capacity observation | About 170K KV headroom was observed | Initialization issue observed | Capacity shortfall observed |
| CUDA Graph | Observed in the experimental route | Fallback issue observed | Admission/space limit observed |
| Fast prefill | FlashInfer path explored | Experimental | Experimental |
| Multimodal serving | No validated SM75 preset | No validated SM75 preset | No validated SM75 preset |

Do not reuse these observations as profile defaults. Kernel selection, memory
budgeting, model runner behavior, and speculative decoding changed between the
legacy cu128 branch and this cu130 migration.

## Multimodal Scope

Gemma4 multimodal model forms are recognized upstream, including image and
other processor-mediated input paths. This fork has not validated an SM75
`0.2.1-pre` text-plus-image, video, or audio deployment preset. Multimodal
requests also introduce encoder behavior that must be tested independently of
text-only decoder CUDA Graph results. See the upstream
[multimodal CUDA Graph notes](design/cuda_graphs_multimodal.md) before
attempting a route.

## Promotion Criteria

A Gemma4 route may move into the main tested-checkpoint list only after it has
all of the following evidence on dual RTX 2080 Ti 22 GB with TP=2:

1. A fresh CUDA 13.0 source build with SM75 kernels.
2. Non-eager target-only generation with a quality probe.
3. CUDA Graph capture and stable repeated requests.
4. Reproducible cold 4K/128 prefill/decode figures with the exact KV dtype and
   MTP configuration recorded.
5. For multimodal or MTP, separate evidence for the encoder or assistant path.

Until then, call Gemma4 baseline-supported or experimental, never a promoted
SM75 deployment route.
