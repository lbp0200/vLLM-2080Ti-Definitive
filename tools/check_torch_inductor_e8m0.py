#!/usr/bin/env python3
"""Exercise the PyTorch E8M0 nested-compile regression fixed by #189530."""

import torch
from torch._higher_order_ops.invoke_subgraph import nested_compile_region


@nested_compile_region
def mutate_input(x, y, scale_e8m0):
    x.add_(1)
    return torch.mul(x, y) * scale_e8m0.to(torch.float32)


def forward(x, y, scale_e8m0):
    return mutate_input(x, y, scale_e8m0) + mutate_input(x, y, scale_e8m0)


def main() -> int:
    x = torch.randn(8)
    y = torch.randn(8)
    scale = torch.full((8,), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    compiled = torch.compile(forward, backend="inductor", fullgraph=True)

    expected_input = x.clone()
    expected = forward(expected_input, y, scale)
    actual = compiled(x, y, scale)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, expected_input)
    print("PyTorch E8M0 functionalization check: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
