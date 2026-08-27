# Issue #112 / #135 validation - 2026-08-27

## Issue #112

Issue #112 is an open SM75/v0.2.x test collection rather than a single bug.
Its core Qwen3.8 NVFP4/no-MTP/non-eager concurrent route is validated on `.31`:

- two RTX 2080 Ti, pinned by UUID (`GPU-4da87347-023e-7014-a837-6b1fb649a42e`, `GPU-1c2b8831-227f-96d1-0c0f-92a189726072`)
- TP2, no MTP, `enforce_eager=False`
- Marlin NVFP4, FlashQLA legacy GDN prefill, FP8 KV
- exact 4096-token prompt / 128-token generation, C1/C2/C4/C8
- CUDA Graph capture and stable replay, all requests successful

The completed C1/C2/C4/C8 results are recorded in
`/mnt/ai-cache/report/qwen38-27b-nvfp4-tp2-concurrency-20260826/concurrency-4k128-final-cli-r3.json`.
The open collection still has uncompleted optional lanes such as MTP3, long
context, and multimodal coverage, so the issue as a whole remains a test
collection rather than a fully closed item.

## Issue #135

The fix is present in `vllm/v1/attention/backends/turboquant_attn.py`:

- `_TQ_FI_PREFILL_WRAPPERS` is an `OrderedDict` LRU cache.
- Default maximum is 16 wrappers, configurable with
  `VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE_MAXSIZE`.
- Eviction happens before planning the replacement, avoiding a temporary
  `maxsize + 1` workspace peak.
- CUDA-graph-safe mode intentionally disables eviction and warns; this remains
  a separate lifecycle caveat.

Targeted tests on `.31`:

```text
2 passed, 143 deselected
```

The real long-context test used the fixed pre3 runtime on `.31`, Qwen3.8-27B
FP8, TurboQuant K8V4, prefix-combine `auto`, TP2, no MTP, and no eager mode.
It sent one conversation from 20,012 prompt tokens to 85,894 prompt tokens in
17 sequential requests. Every request succeeded. The log shows the intended
prefix-combine path with `cached_len` advancing from 18,720 to 85,280. GPU
free memory changed from 3,041 MiB before the first request to 2,589 MiB after
the last request, then returned to 8 MiB driver baseline after shutdown; there
was no OOM, `EngineDeadError`, or GDN temporary-allocation failure.

Artifacts:

- `/mnt/ai-cache/report/issue135-longctx-20260827/issue135-longctx-20260827.json`
- `/mnt/ai-cache/report/issue135-longctx-20260827/remote-logs/`

Conclusion: the default (non CUDA-graph-safe) #135 wrapper leak is fixed and
passes both targeted and real long-context validation. The
`VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE=1` path still needs an
independent bounded-lifetime design before it can be called fully resolved.

## Dependency follow-up from #112

The Minachist Qwen3.8 INT6 loading report was caused by an old
`humming-kernels` requirement. `requirements/cuda.txt` now pins
`humming-kernels[cu13]==0.1.13`, which is the first version reported to fix the
AutoRound/INT6 path. The `.31` runtime environment was upgraded to the same
version and its Humming quantization/NVFP4 modules import successfully. A full
Minachist model load was not rerun because that checkpoint is not present on
the test host. The `.31` vLLM editable dist-info still reports the old `0.1.10`
requirement because refreshing the editable install was blocked by missing
`setuptools_rust`; a clean rebuild should refresh that metadata.
