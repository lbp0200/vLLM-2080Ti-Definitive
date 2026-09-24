# 非交互启动

先执行 `./build.sh`，再传入扁平化的 profile 路径。Profile 只包含路线参数；检查点、GPU 拓扑、端口和 mode 都由 launcher 管理。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` 可省略，默认是 `fast`。没有 `MODE` 的 profile 会保留 launcher 的选择；profile 中显式的 `MODE=normal` 或 `MODE=fast` 可以覆盖它。目录中不再区分 `fast/` 和 `normal/`。

可以使用 `--model-dir`、`--speculative-model`、`--profile`、`--mode`、`--gpu-devices`、`--tp-size`、`--pp-size`、`--port`、`--start-timeout` 和 `--print-config`。高级 launcher/runtime 参数使用 `--set KEY=VALUE`。不要把 Prefix Cache、Mamba cache、GPU、端口或模型路径写入 profile，验证器会拒绝这些字段。

Profile 库是验证矩阵，不代表每个文件在每台机器上都能运行。只有外部审计完成启动、4K/128、32K/512、并发和图文正确性验证后，路线才会被 promote。
