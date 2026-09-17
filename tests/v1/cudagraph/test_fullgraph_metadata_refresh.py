# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    _copy_fullgraph_metadata,
)


@dataclass
class _Metadata:
    num_spec_decodes: int
    positions: torch.Tensor
    nested: dict[str, torch.Tensor]


def test_copy_fullgraph_metadata_preserves_captured_storage():
    captured = _Metadata(
        num_spec_decodes=1,
        positions=torch.tensor([1, 2], dtype=torch.int32),
        nested={"block_table": torch.tensor([[0, 1]], dtype=torch.int32)},
    )
    positions_ptr = captured.positions.data_ptr()
    block_table_ptr = captured.nested["block_table"].data_ptr()
    runtime = _Metadata(
        num_spec_decodes=1,
        positions=torch.tensor([7, 8], dtype=torch.int32),
        nested={"block_table": torch.tensor([[4, 5]], dtype=torch.int32)},
    )

    _copy_fullgraph_metadata(runtime, captured)

    assert captured.positions.data_ptr() == positions_ptr
    assert captured.nested["block_table"].data_ptr() == block_table_ptr
    torch.testing.assert_close(captured.positions, runtime.positions)
    torch.testing.assert_close(
        captured.nested["block_table"], runtime.nested["block_table"]
    )


def test_copy_fullgraph_metadata_rejects_topology_change():
    captured = _Metadata(
        num_spec_decodes=1,
        positions=torch.tensor([1], dtype=torch.int32),
        nested={},
    )
    runtime = _Metadata(
        num_spec_decodes=2,
        positions=torch.tensor([1], dtype=torch.int32),
        nested={},
    )

    with pytest.raises(RuntimeError, match="num_spec_decodes"):
        _copy_fullgraph_metadata(runtime, captured)


def test_v2_fullgraph_stages_explicit_metadata_before_replay(monkeypatch):
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=1,
        uniform_token_count=8,
    )
    captured = _Metadata(
        num_spec_decodes=1,
        positions=torch.tensor([1], dtype=torch.int32),
        nested={},
    )
    captured_slots = {"layer": torch.tensor([2], dtype=torch.int64)}
    replay = SimpleNamespace(calls=0)

    def _replay():
        replay.calls += 1

    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager.graphs = {desc: SimpleNamespace(replay=_replay)}
    manager._fullgraph_attn_metadata = {desc: captured}
    manager._fullgraph_slot_mappings = {desc: captured_slots}
    monkeypatch.setattr(
        cudagraph_utils,
        "get_offloader",
        lambda: SimpleNamespace(sync_prev_onload=lambda: None),
    )

    manager.run_fullgraph(
        desc,
        attn_metadata=_Metadata(
            num_spec_decodes=1,
            positions=torch.tensor([9], dtype=torch.int32),
            nested={},
        ),
        slot_mapping={"layer": torch.tensor([6], dtype=torch.int64)},
    )

    assert replay.calls == 1
    assert captured.positions.item() == 9
    assert captured_slots["layer"].item() == 6
