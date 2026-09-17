# Changelog

This changelog tracks releases of vLLM 2080 Ti Definitive Edition separately from upstream vLLM releases.

## v0.2.1-RC - 2026-09-18

Release candidate for the CUDA 13 / PyTorch 2.13 SM75 line.

- Adds validated DFlash2 routes, fused Triton grouped convolution, stable SM75
  FlashInfer FA2 graph buffers, and live metadata refresh for FULL graph replay.
- Fixes concurrent speculative graph sizing, automatic prefill alignment,
  independent target/draft KV pools, and concurrent MTP FA2 execution.
- Expands the launcher with DFlash2 controls and reproducible 4K/128 plus
  32K/512 startup benchmarks.
- Generalizes qualified PCIe custom all-reduce to 2–16 GPUs and improves the
  CUDA 13 Torch/Triton mirror build path.
- Converges profiles on validated routes, requires explicit KV precision, and
  removes unsupported concurrency presets.
- Reworks the bilingual README and launcher guides while removing stale reports
  and redundant documentation.

## v0.2.1-pre4 - 2026-09-14

Nightly-alignment candidate based on upstream `b23433088b`
(`v0.29.1rc0-33`).

- Adopts upstream replacements for previously local runtime fixes and retains
  only the SM75-specific FlashQLA, TurboQuant, parser, and rendezvous paths.
- Keeps `humming-kernels[cu13]==0.1.13` for INT6/AutoRound loading.
- Trims unrelated upstream documentation and automation from the fork.

## v0.2.1-pre3 - 2026-08-27

Validated true-concurrency candidate for dual RTX 2080 Ti.

- Adds packed-varlen FlashQLA, prefill batch alignment, bounded TurboQuant
  wrapper caching, and long-context workspace handling.
- Improves reasoning/tool parsing, Mamba/offload boundaries, GPU selection, and
  single-node process rendezvous.
- Validates the documented NVFP4 concurrency route and the associated parser,
  serving, TurboQuant, offload, and FlashQLA paths.

## v0.2.1-pre2 - 2026-08-21

First validated preview based on upstream vLLM `v0.27.1`.

- Adds Qwen3.8 FP8/NVFP4 profiles, TP/PP launcher selection, stricter preflight
  checks, and the required SM75 FlashInfer/FlashQLA compatibility paths.
- Ports the validated reasoning, tool-call, CUDA Graph, and TurboQuant fixes to
  the newer runtime architecture.
- Validates the documented dual-2080-Ti TP=2 routes; broader PP and additional
  long-context routes remained experimental.

## v0.2.1-pre

Initial CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 migration snapshot based
on upstream vLLM `v0.27.1`.

Thanks to upstream vLLM and its contributors. Fork release and validation work
is maintained by [@weicj](https://github.com/weicj), with contributions from
the project community.
