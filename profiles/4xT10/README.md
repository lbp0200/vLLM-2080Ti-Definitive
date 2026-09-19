# 4xT10 Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x256k-text-only.env` | 256K | FP16 | DFlash2/7 | text-only | 300,304 | 1433.91 / 191.89 | 1536.13 / 189.38 |
| `qwen27b/w8a16/dflash2-fp16kv-1x235k-text-image.env` | 235K | FP16 | DFlash2/7 | text+image | 241,215 | 1443.96 / 190.89 | 1532.96 / 187.85 |
| `qwen27b/w8a16/mtp4-fp16kv-1x256k-text-image.env` | 256K | FP16 | MTP/4 | text+image | 320,484 | 1484.30 / 106.04 | 1527.71 / 104.25 |
| `qwen27b/w8a16/mtp3-fp8kv-2x256k-text-image.env` | 2 x 256K | FP8 | MTP/3 | text+image | 621,102 | 1466.79 / 96.05 | 1496.23 / 93.72 |

## Notes

1. Profiles are flat under each model/weight directory. Mode is selected by the launcher and defaults to `fast`.
2. Performance uses the launcher's reproducible reference lane: Prefix Cache is disabled only during testing, one text-only request is sent at a time, warm-up is excluded, 4K/128 is the median of three requests, and 32K/512 is run to completion. Multimodal profiles use the same text-only performance lane; image semantics are validated separately. `4K/128` means exactly 4,096 input tokens and 128 output tokens; `32K/512` means exactly 32,768 input tokens and 512 output tokens. Raw logs and request JSON remain in the external audit directory.
3. Test environment: validated on 2026-09-19 with v0.2.1. The host has two sockets with Intel Xeon E5-2673 v4 CPUs (80 logical CPUs total), 60 GiB RAM and 8 GiB swap. The tested topology uses physical GPUs 0, 2, 3 and 4, four Tesla T10 GPUs with 16,384 MiB each; all four are on the same NUMA node and connected by PCIe PIX links, and the service uses TP4 over this group. The host driver is NVIDIA 595.91.07. The runtime is the repository's `vllm-sm75-tp2-cu130` environment (CUDA 13.0, torch 2.13.0+cu130).
