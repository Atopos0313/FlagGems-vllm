# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import pytest
import torch

import flaggems_vllm

from . import base, consts

try:
    from vllm.v1.worker.gpu.sample.logit_bias import (
        apply_logit_bias as vllm_apply_logit_bias,
    )

    _VLLM_PARAMS = set(inspect.signature(vllm_apply_logit_bias).parameters)
except Exception:  # vLLM missing or too old to ship the v1 GPU sampler
    vllm_apply_logit_bias = None
    _VLLM_PARAMS = set()

# Per-request table widths of vLLM's LogitBiasState (MAX_NUM_ALLOWED_TOKEN_IDS,
# MAX_NUM_LOGIT_BIAS_TOKENS, MAX_NUM_STOP_TOKEN_IDS); they make BLOCK_SIZE=1024.
W_ALLOWED, W_BIAS, W_STOP = 1024, 1024, 128
# Rows of the per-request tables (scheduler max_num_seqs); requests sit in
# scattered slots and reach them through expanded_idx_mapping.
MAX_NUM_REQS = 1024
NUM_ALLOWED, NUM_BIAS, NUM_STOP = 256, 16, 4
# pos + 1 < MIN_LEN for every token, so min_tokens masking is active.
BASE_POS, MIN_LEN = 8, 64
# 131 is coprime to every benchmarked vocab size, so strided ids are unique.
ID_STRIDE = 131

# Each scenario cycles a per-request (allowed, logit_bias, min_tokens) pattern.
# The paths have very different costs: logit_bias / min_tokens touch a few
# entries per row (launch bound), allowed_token_ids rewrites the whole row and
# the restore path scans the whole row (memory bound).
SCENARIOS = {
    "logit_bias": [(0, 1, 0)],
    "min_tokens": [(0, 0, 1)],
    "allowed_token_ids": [(1, 0, 0)],
    "min_tokens_restore": [(0, 0, 1)],
    "mixed": [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)],
}
RESTORE_SCENARIOS = {"min_tokens_restore"}
FULL_ROW_SCENARIOS = {"allowed_token_ids", "min_tokens_restore", "mixed"}


def _vllm_baseline(logits, **state):
    """Call vLLM with only the arguments its installed version understands."""
    args = dict(state, logits=logits)
    vllm_apply_logit_bias(**{k: v for k, v in args.items() if k in _VLLM_PARAMS})


def _id_table(width, num_rows, vocab_size, device):
    rows = torch.arange(num_rows, device=device).unsqueeze(1) * 997
    cols = torch.arange(width, device=device).unsqueeze(0) * ID_STRIDE
    return ((rows + cols) % vocab_size).to(torch.int32)


class LogitBiasBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_reqs, tokens_per_req, vocab_size"

    def __init__(self, *args, scenario, **kwargs):
        super().__init__(*args, **kwargs)
        # set_metrics() appends to this list in place; keep the shared default
        # untouched so "gbps" does not leak into other scenarios.
        self.metrics = list(self.metrics)
        self.scenario = scenario
        self._full_row_bytes = 0
        self._current_shape = None

    def record_shapes(self, *args, **kwargs):
        # The per-request tables have fixed vLLM widths; only the case matters.
        return list(self._current_shape)

    def set_more_shapes(self):
        return [
            (3, 1, 1000),
            (1, 1, 151936),
            (128, 1, 128256),
            (32, 4, 128256),
            (512, 1, 151936),
            (1024, 1, 32000),
        ]

    def set_more_metrics(self):
        return ["gbps"] if self.scenario in FULL_ROW_SCENARIOS else []

    def get_gbps(self, args, latency=None):
        return self._full_row_bytes * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, dtype):
        device = self.device
        restore = self.scenario in RESTORE_SCENARIOS
        pattern = torch.tensor(SCENARIOS[self.scenario], dtype=torch.int32)
        for num_reqs, tokens_per_req, vocab_size in self.shapes:
            num_tokens = num_reqs * tokens_per_req
            max_num_reqs = max(MAX_NUM_REQS, num_reqs)
            flags = pattern[torch.arange(num_reqs) % pattern.shape[0]].to(device)
            slots = torch.randperm(max_num_reqs, device=device)[:num_reqs]

            def per_slot(values):
                table = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
                table[slots] = values.to(torch.int32)
                return table

            state = dict(
                expanded_idx_mapping=slots.repeat_interleave(tokens_per_req).to(
                    torch.int32
                ),
                pos=(BASE_POS + torch.arange(tokens_per_req, device=device))
                .repeat(num_reqs)
                .to(torch.int32),
                num_allowed_token_ids=per_slot(flags[:, 0] * NUM_ALLOWED),
                allowed_token_ids=_id_table(
                    W_ALLOWED, max_num_reqs, vocab_size, device
                ),
                num_logit_bias=per_slot(flags[:, 1] * NUM_BIAS),
                logit_bias_token_ids=_id_table(
                    W_BIAS, max_num_reqs, vocab_size, device
                ),
                logit_bias=torch.randn(
                    max_num_reqs, W_BIAS, dtype=torch.float32, device=device
                ),
                min_lens=torch.full(
                    (max_num_reqs,), MIN_LEN, dtype=torch.int32, device=device
                ),
                num_stop_token_ids=per_slot(flags[:, 2] * NUM_STOP),
                restore_when_all_masked=per_slot(flags[:, 2] * int(restore)),
                stop_token_ids=_id_table(W_STOP, max_num_reqs, vocab_size, device),
                check_all_masked_rows=restore,
            )

            # Whole-row traffic per call: the allow-list path writes every logit
            # and the restore path reads every logit of an active row.
            full_rows = int(flags[:, 0].sum()) + int(restore) * int(flags[:, 2].sum())
            elem_size = torch.empty((), dtype=dtype).element_size()
            self._full_row_bytes = full_rows * tokens_per_req * vocab_size * elem_size

            # In-place op: timing repetitions keep re-applying it to the same
            # logits, values drift but the memory traffic per call does not.
            logits = torch.randn(num_tokens, vocab_size, dtype=dtype, device=device)
            self._current_shape = (num_reqs, tokens_per_req, vocab_size)
            yield logits, state


def _run_with_vllm_label(bench):
    """Run `bench` with the baseline columns labelled vLLM instead of Torch."""
    original_str = base.BenchmarkResult.__str__

    def vllm_str(result):
        # Same width as the replaced headers so the table stays aligned.
        return (
            original_str(result)
            .replace("Torch Latency (ms)", "vLLM  Latency (ms)")
            .replace("Torch GBPS ", "vLLM  GBPS ")
        )

    base.BenchmarkResult.__str__ = vllm_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


@pytest.mark.apply_logit_bias
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(vllm_apply_logit_bias is None, reason="need vLLM v1 GPU sampler")
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_apply_logit_bias(scenario):
    if scenario in RESTORE_SCENARIOS and "restore_when_all_masked" not in _VLLM_PARAMS:
        pytest.skip("installed vLLM has no restore_when_all_masked support")
    bench = LogitBiasBenchmark(
        op_name=f"apply_logit_bias_{scenario}",
        torch_op=_vllm_baseline,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        scenario=scenario,
    )
    bench.set_gems(flaggems_vllm.apply_logit_bias)
    print(
        "\n[baseline] vLLM's own apply_logit_bias from "
        "vllm.v1.worker.gpu.sample.logit_bias; the speedup column is vs vLLM."
    )
    _run_with_vllm_label(bench)
