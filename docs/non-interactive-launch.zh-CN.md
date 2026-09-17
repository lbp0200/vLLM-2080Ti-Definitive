# 非交互启动

[English](non-interactive-launch.md)

非交互模式无需进入 Launcher 菜单即可按给定配置启动 vLLM 服务。使用前请先运行
`./build.sh` 完成构建。

## 使用 Profile 启动

传入模型目录、相对于 `profiles/` 的 profile 路径、目标 GPU 和匹配的启动模式：

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env \
  --mode normal \
  --gpu-devices 4,5 \
  --tp-size 2 \
  --pp-size 1
```

传入任意有效配置参数后会自动进入非交互模式，因此上述写法无需额外添加
`--non-interactive`。Launcher 会等待 `/health` 就绪、执行 smoke 请求，并输出 API
地址、PID 文件和日志文件路径。

## 启动前预览

在相同命令后添加 `--print-config`，可以解析 profile 和全部覆盖参数，但不启动服务：

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env \
  --mode normal \
  --gpu-devices 4,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

## 常用参数

| 参数 | 用途 | 示例 |
| --- | --- | --- |
| `--model-dir` | 模型权重目录 | `/mnt/models/Qwen3.8-27B-FP8` |
| `--profile` | 相对于 `profiles/` 的路线预设 | `2x2080Ti/qwen27b/w8a16/...env` |
| `--mode` | `safe`、`normal`、`fast` 或 `aggressive` | `fast` |
| `--gpu-devices` | 物理 GPU 编号 | `4,5` |
| `--tp-size` / `--pp-size` | TP/PP 布局 | `2` / `1` |
| `--port` | API 端口 | `8000` |
| `--service-scope` | `local` 或 `lan` | `local` |
| `--start-timeout` | 启动超时秒数 | `900` |
| `--print-config` | 打印最终配置后退出 | 无参数值 |

Profile 中的参数可通过对应的 lower-kebab-case 选项覆盖。高级 Launcher 或
`VLLM_*` 参数使用 `--set KEY=VALUE`；使用 `--unset KEY` 可清除继承值并回到
Launcher 默认值。

```bash
./launcher.sh \
  --model-dir /path/to/checkpoint \
  --profile 2x2080Ti/qwen27b/w4a16/fast/dflash2-tqk8v4-1x256k-text-image.env \
  --mode fast \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --port 18080 \
  --set HF_DOWNLOAD_ROUTE_MODE=official
```

配置优先级为 `CLI > 已导出的环境变量 > profile > 默认值`。布尔参数接受
`0/1`；不带参数值的布尔选项等同于 `1`。

## 环境变量写法

也可以使用等价的环境变量形式：

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=2x2080Ti/qwen27b/w8a16/normal/mtp-fp8kv-1x256k-text-only.env \
MODE=normal \
GPU_DEVICES=4,5 \
TP_SIZE=2 \
PP_SIZE=1 \
NON_INTERACTIVE=1 \
./launcher.sh
```

自动化部署优先推荐 CLI 参数写法，因为有效覆盖项可以直接从命令中确认。

## 日志与停止服务

日志、PID 文件和 Launcher 状态默认写入 `run-logs/`。启动成功后 Launcher 会输出
确切路径。停止托管服务时运行 `./launcher.sh` 并选择 **Stop service**；当前没有
非交互停止参数。
