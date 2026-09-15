#!/usr/bin/env python3
"""Backport the PyTorch E8M0 auto-functionalization fix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FIX = """    # auto_functionalized wrappers are never lowered directly; they must always
    # be processed by decompose_auto_functionalized (its trailing scan asserts
    # they were removed), so unsupported input types must not skip them here.
    if node.target in (
        torch.ops.higher_order.auto_functionalized,
        torch.ops.higher_order.auto_functionalized_v2,
    ):
        return False

"""
FUNCTION = (
    "def fallback_node_due_to_unsupported_type("
    "node: torch.fx.Node, allow_cpu_inputs=True):\n"
)
FIRST_STATEMENT = """    # Custom fallback lowering
"""


def patch_lowering(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if FIX in source:
        return False

    function_count = source.count(FUNCTION)
    if function_count != 1:
        raise ValueError(
            f"expected one fallback_node_due_to_unsupported_type definition, "
            f"found {function_count}"
        )

    insertion_point = FUNCTION + FIRST_STATEMENT
    if insertion_point not in source:
        raise ValueError(
            "PyTorch lowering.py does not match the validated 2.13.0 layout"
        )

    patched = source.replace(
        insertion_point,
        FUNCTION + FIX + FIRST_STATEMENT,
        1,
    )
    path.write_text(patched, encoding="utf-8")
    return True


def find_torch_lowering() -> Path:
    spec = importlib.util.find_spec("torch._inductor.lowering")
    if spec is None or spec.origin is None:
        raise ValueError("cannot locate torch._inductor.lowering")
    return Path(spec.origin).resolve()


def main() -> int:
    if len(sys.argv) > 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [lowering.py]")

    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else find_torch_lowering()
    try:
        changed = patch_lowering(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot patch PyTorch Inductor at {path}: {exc}") from exc

    state = "applied" if changed else "already applied"
    print(f"PyTorch E8M0 functionalization fix: {state} ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
