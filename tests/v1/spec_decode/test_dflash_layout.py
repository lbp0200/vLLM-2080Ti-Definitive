# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.speculator import (
    _pad_uniform_query_start_loc,
)


def test_full_graph_padding_keeps_uniform_query_rows_distinct():
    starts = _pad_uniform_query_start_loc(np.array([0, 8], dtype=np.int32), 1, 8)

    torch.testing.assert_close(
        starts,
        torch.tensor([0, 8, 16, 24, 32, 40, 48, 56, 64], dtype=torch.int32),
    )


def test_full_graph_padding_preserves_live_rows():
    starts = _pad_uniform_query_start_loc(
        np.array([0, 8, 16], dtype=np.int32), 2, 4
    )

    torch.testing.assert_close(
        starts,
        torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32),
    )
