# 2xT10 Profile

## Qwen3.8-27B-INT4

Tested weight: [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/mtp3-fp8kv-1x262K-text-only.env` | 262K | FP8 | MTP/3 | text-only | 266,537 | 1122.00 / 87.88 | 1070.10 / 83.57 |
| `qwen27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env` | 2 x 155K | TQK8V4 | MTP/3 | text-only | 321,789 | 1138.22 / 87.42 | 1000.18 / 59.55 |
| `qwen27b/w4a16/mtp3-tq4nc-2x220K-text-only.env` | 2 x 220K | TQ4NC | MTP/3 | text-only | 471,807 | 1134.92 / 83.54 | 1100.51 / 49.76 |
| `qwen27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env` | 196K | TQK8V4 | MTP/3 | text+image | 206,888 | 1138.74 / 87.46 | 1053.94 / 50.20 |

## Notes

1. Profiles are flat under the model/weight directory. Mode is selected by the launcher and defaults to `fast`.
2. Performance uses the launcher's reproducible reference lane: Prefix Cache is disabled only during testing, one text-only request is sent at a time, warm-up is excluded, 4K/128 is the median of three requests, and 32K/512 is run to completion. Multimodal profiles use the same text-only performance lane; image semantics are validated separately. `4K/128` means exactly 4,096 input tokens and 128 output tokens; `32K/512` means exactly 32,768 input tokens and 512 output tokens.
3. Test environment: validated on 2026-09-25 with the v0.2.2 runtime development build. The host has two sockets with Intel Xeon E5-2673 v4 CPUs (80 logical CPUs total), 60 GiB RAM and 8 GiB swap. The tested topology uses physical GPUs 0 and 2, two Tesla T10 GPUs with 16,384 MiB each; both are on the same NUMA node and connected by PCIe PIX, and the service uses TP2. The NVIDIA driver is 595.91.07. The runtime uses CUDA 13.0 and torch 2.13.0+cu130.
4. The multimodal route started with the vision path enabled at 196K and correctly described a JPEG flowchart's proxy/prefill/decode pipeline.
