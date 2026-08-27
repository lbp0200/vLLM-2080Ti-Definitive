# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device(DEVICE)

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


def test_packed_decode_keeps_beta_in_fp32():
    device = torch.device(DEVICE)
    dtype = torch.bfloat16

    mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
    a = torch.zeros((1, 1), device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros((1,), device=device, dtype=dtype)
    ssm_state_indices = torch.ones((1,), device=device, dtype=torch.int32)

    state_packed = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
    out_packed = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
    )

    expected_beta = torch.sigmoid(b.float()).squeeze()
    torch.testing.assert_close(
        state_packed[1, 0, 0, 0], expected_beta, rtol=1e-6, atol=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_fused_recurrent_packed_decode_accepts_gdn_state_slot_zero() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float16
    batch_size = 2
    num_heads = value_heads = 1
    key_dim = value_dim = 16
    qkv_dim = 2 * num_heads * key_dim + value_heads * value_dim

    mixed_qkv = torch.randn(batch_size, qkv_dim, device=device, dtype=dtype)
    mixed_qkv[1].copy_(mixed_qkv[0])
    a = torch.randn(batch_size, value_heads, device=device, dtype=dtype)
    a[1].copy_(a[0])
    b = torch.randn(batch_size, value_heads, device=device, dtype=dtype)
    b[1].copy_(b[0])
    a_log = torch.randn(value_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, device=device, dtype=dtype)
    initial_state = torch.randn(
        2, value_heads, value_dim, key_dim, device=device, dtype=dtype
    )
    initial_state[1].copy_(initial_state[0])

    state = initial_state.clone()
    out = torch.empty(
        batch_size, 1, value_heads, value_dim, device=device, dtype=dtype
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=torch.tensor([0, 1], device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        null_block_id=PAD_SLOT_ID,
    )

    torch.testing.assert_close(out[0], out[1], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(state[0], state[1], rtol=2e-3, atol=2e-3)

    padded_state = initial_state.clone()
    padded_out = torch.empty_like(out)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=padded_state,
        out=padded_out,
        ssm_state_indices=torch.tensor(
            [0, PAD_SLOT_ID], device=device, dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
        null_block_id=PAD_SLOT_ID,
    )

    torch.testing.assert_close(padded_out[0], out[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(padded_out[1], torch.zeros_like(padded_out[1]))
    torch.testing.assert_close(padded_state[0], state[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(padded_state[1], initial_state[1], rtol=0, atol=0)


def test_packed_decode_supports_large_batch_head_grid():
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = torch.device(DEVICE)
    gates = torch.empty((B, HV), device=device)
    params = torch.empty((HV,), device=device)
    out = torch.empty((B, 1, HV, V), device=device)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.empty((B, 2 * H * K + HV * V), device=device),
        a=gates,
        b=gates,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=torch.empty((1, HV, V, K), device=device),
        out=out,
        ssm_state_indices=torch.zeros((B,), device=device, dtype=torch.int32),
    )

    assert torch.count_nonzero(out).item() == 0
