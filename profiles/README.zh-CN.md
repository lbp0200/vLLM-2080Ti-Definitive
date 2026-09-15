# Profile 导引

语言：[English](README.md) | 简体中文

Profile 是只保存路线参数的 `.env` 预设，不负责选择 checkpoint、GPU、端口、chat
template 或 reasoning 默认值；模型路径通过 `MODEL_DIR` 单独指定。

Profile 首先按硬件分组，再按模型、权重格式和启动模式分层：

```text
profiles/
  2x2080Ti/   # [硬件说明](2x2080Ti/README.zh-CN.md)
  4xT10/      # [硬件说明](4xT10/README.zh-CN.md)
```

Profile 文件名统一使用 `<解码类型>-<KV精度>-<并发数><上下文>-<消息类型>.env`。
例如 `dflash2-tqk8v4-2x172k-text-only.env` 表示 DFlash2、TQK8V4 KV、双并发、
每路 172K 上下文、纯文本消息。`MTP_K=0` 表示不使用 MTP，`MTP_K=3` 表示 MTP3。

选择 profile 后，可执行 `./launcher.sh --print-config` 检查最终生效的路线参数。
