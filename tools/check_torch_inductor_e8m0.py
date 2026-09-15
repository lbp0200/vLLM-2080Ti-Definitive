#!/usr/bin/env python3
"""Verify the PyTorch 2.13 Inductor E8M0 backport is installed."""

import inspect

import torch
import torch._inductor.lowering as lowering


def main() -> int:
    source = inspect.getsource(lowering.fallback_node_due_to_unsupported_type)
    if "auto_functionalized" not in source or "auto_functionalized_v2" not in source:
        raise SystemExit("PyTorch Inductor E8M0 functionalization fix is missing")
    print(f"PyTorch E8M0 functionalization check: passed ({torch.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
