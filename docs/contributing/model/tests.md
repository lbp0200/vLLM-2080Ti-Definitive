# Model Tests

Model changes in this fork require a focused correctness probe and, when they affect serving or scheduling, a non-eager CUDA Graph smoke test on the supported SM75 route. Record the exact checkpoint, quantization, KV precision, MTP setting, context shape, and output-quality result before promoting a profile.

Use the nearest existing parser, model, or serving test and keep hardware benchmarks under `benchmarks/`. A model that only imports or loads is not evidence of a supported deployment route.
