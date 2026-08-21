# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_non_final_pp_rank_does_not_report_a_drafter():
    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = object()

    assert not runner._is_drafter_rank()

    runner.drafter = object()

    assert runner._is_drafter_rank()
