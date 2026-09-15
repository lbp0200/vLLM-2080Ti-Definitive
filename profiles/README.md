# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Profiles are `.env` presets for route parameters. They do not select the
checkpoint, GPUs, port, chat template, or reasoning defaults; those remain
launcher settings. Select the model path separately with `MODEL_DIR`.

Profiles are grouped by hardware first, then model family, weight format, and
startup mode:

```text
profiles/
  2x2080Ti/   # [hardware-specific guide](2x2080Ti/README.md)
  4xT10/      # [hardware-specific guide](4xT10/README.md)
```

Profile filenames use `<decoder>-<kv>-<concurrency><context>-<message>.env`.
For example, `dflash2-tqk8v4-2x172k-text-only.env` is a DFlash2 route using
TQK8V4 KV, two concurrent requests, 172K context per request, and text-only
messages. `MTP_K=0` means no MTP; `MTP_K=3` means MTP3.

Use `./launcher.sh --print-config` after selecting a profile to inspect the
resolved route before starting the service.
