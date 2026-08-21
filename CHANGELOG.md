# Changelog

This changelog tracks releases of vLLM 2080 Ti Definitive Edition separately from upstream vLLM releases.

## v0.2.1-pre2 - 2026-08-21

`v0.2.1-pre2` is the second public preview of the CUDA 13 based 0.2.x line. The maintained 0.1.x line remains the project's primary stable release line; 0.2.x is intended for testing the newer upstream runtime and must not yet be treated as a replacement.

### Changes Since v0.2.1-pre

- Moves the migration snapshot to an evidence-backed preview based on upstream vLLM `v0.27.1`, CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 source builds.
- Adds profile-driven Qwen3.8 FP8 and NVFP4 routes, an improved interactive launcher, explicit TP/PP layout selection, physical GPU ordering, and stricter build/runtime preflight checks.
- Ports the validated SM75 fixes for Qwen reasoning, named and streaming tool calls, CUDA Graph profiling, TurboQuant workspace reservation, and TurboQuant decode diagnostics to the newer runtime architecture.
- Restores the FlashInfer and FlashQLA legacy paths required by the validated SM75 runtime without importing TileLang during normal model initialization.
- Adds bilingual profile documentation, a validation report, and a [PR migration audit](docs/0.2.x-pr-migration-audit.md) that separates migrated, absorbed, experimental, and unsupported work.

### Validation

- The official Qwen3.8-27B-FP8 checkpoint passed non-eager TP=2/PP=1 CUDA Graph startup and output checks on dual RTX 2080 Ti with FP16 KV and TurboQuant K8V4 routes.
- In the controlled 4K/128 FP16-KV comparison, pre2 measured `1570.57 / 30.01` prefill/decode tok/s without MTP and `1522.72 / 75.83` with MTP3. The corresponding 0.1.x baseline measurements were `1420.07 / 30.01` and `1453.46 / 60.70`.
- Profile validation, GDN slot handling, Mamba align, CPU KV offload, reasoning, tool-call, and serving regressions passed for the documented paths.

### Known Limitations

- Promoted serving support is limited to the documented TP=2/PP=1 profiles. PP greater than one, PP with MTP, and PP with DSpark/DFlash/EAGLE3 are not promoted in pre2.
- TurboQuant long-context prefix-combine and retained Qwen3.x 35B profiles require additional CUDA 13 target-GPU validation before promotion.
- MXFP4 is not supported on SM75 by the current upstream Marlin kernels; NVFP4 and MXFP4 are different formats.

### Credits

Thanks to upstream vLLM and its contributors. Fork release and validation work is maintained by [@weicj](https://github.com/weicj), with contributions migrated or evaluated from [@YuYue1208](https://github.com/YuYue1208), [@superniker](https://github.com/superniker), [@kevinhirsch](https://github.com/kevinhirsch), [@hotwa](https://github.com/hotwa), and [@0xYYP](https://github.com/0xYYP).

## v0.2.1-pre

Initial CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 migration snapshot based on upstream vLLM `v0.27.1`.
