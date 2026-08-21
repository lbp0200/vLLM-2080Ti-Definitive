# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.qwen3_5_mtp import (
    Qwen3_5MTP,
    Qwen3_5MultiTokenPredictor,
)


class _DraftLayer(nn.Module):
    use_attn_reduce_scatter_for_moe = False

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, positions, hidden_states, residual):
        self.calls += 1
        return hidden_states + 1, residual


class _FinalNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, None


def test_qwen3_5_mtp_declares_pipeline_parallel_support():
    assert supports_pp(Qwen3_5MTP)


def test_predictor_executes_complete_draft_layer_locally(monkeypatch):
    predictor = object.__new__(Qwen3_5MultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.do_not_compile = True
    predictor.config = SimpleNamespace(hidden_size=4)
    predictor.num_mtp_layers = 1
    predictor.embed_tokens = nn.Embedding(16, 4)
    predictor.pre_fc_norm_embedding = nn.Identity()
    predictor.pre_fc_norm_hidden = nn.Identity()
    predictor.fc = nn.Linear(8, 4, bias=False)
    draft_layer = _DraftLayer()
    predictor.layers = nn.ModuleList([draft_layer])
    predictor.norm = _FinalNorm()

    class _LastOnlyPPGroup:
        is_first_rank = False
        is_last_rank = True

    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
        lambda: _LastOnlyPPGroup(),
    )

    output = predictor(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        hidden_states=torch.ones(2, 4),
    )

    assert output.shape == (2, 4)
    assert draft_layer.calls == 1


def test_qwen3_5_mtp_forwards_spec_step_idx():
    mtp = object.__new__(Qwen3_5MTP)
    nn.Module.__init__(mtp)
    mtp.do_not_compile = True

    class _RecordingPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec_step_idx = None

        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        ):
            self.spec_step_idx = spec_step_idx
            return hidden_states

    mtp.model = _RecordingPredictor()
    hidden_states = torch.ones(2, 4)
    output = mtp(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        hidden_states=hidden_states,
        spec_step_idx=2,
    )

    assert output is hidden_states
    assert mtp.model.spec_step_idx == 2
