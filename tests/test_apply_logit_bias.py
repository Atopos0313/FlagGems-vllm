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

from . import accuracy_utils as utils
from . import conftest as cfg

try:
    from vllm.v1.worker.gpu.sample.logit_bias import (
        apply_logit_bias as vllm_apply_logit_bias,
    )

    _VLLM_PARAMS = set(inspect.signature(vllm_apply_logit_bias).parameters)
except Exception:  # vLLM missing or too old to ship the v1 GPU sampler
    vllm_apply_logit_bias = None
    _VLLM_PARAMS = set()

# Newer vLLM (>= commit 947dd62) can restore stop tokens when a structured-output
# row would otherwise become all -inf; older releases do not have the argument.
_VLLM_HAS_RESTORE = "restore_when_all_masked" in _VLLM_PARAMS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="apply_logit_bias needs CUDA"
)

if cfg.QUICK_MODE:
    SHAPES = [(8, 1000)]
    WIDTHS = [(16, 16, 8)]
    DTYPES = [torch.float32]
else:
    SHAPES = [(1, 128), (7, 1000), (32, 32000), (5, 151936)]
    # (allowed, logit_bias, stop) table widths; (1024, 1024, 128) mirrors vLLM maxima.
    WIDTHS = [(16, 16, 8), (1024, 1024, 128)]
    DTYPES = utils.FLOAT_DTYPES

STOP_TOKEN = 3
OTHER_TOKEN = 4
PROMPT_LEN = 2
MIN_TOKENS = 10


def _unique_ids(gen, vocab_size, count):
    return torch.randperm(vocab_size, generator=gen)[:count].to(torch.int32)


def _make_state(
    num_tokens, vocab_size, widths, seed, max_reqs=6, with_restore_flags=False
):
    """Random per-request tables + per-token mapping, built on CPU then moved."""
    w_allowed, w_bias, w_stop = widths
    gen = torch.Generator().manual_seed(seed)

    num_allowed = torch.zeros(max_reqs, dtype=torch.int32)
    allowed = torch.zeros(max_reqs, w_allowed, dtype=torch.int32)
    num_bias = torch.zeros(max_reqs, dtype=torch.int32)
    bias_ids = torch.zeros(max_reqs, w_bias, dtype=torch.int32)
    bias = torch.zeros(max_reqs, w_bias, dtype=torch.float32)
    min_lens = torch.zeros(max_reqs, dtype=torch.int32)
    num_stop = torch.zeros(max_reqs, dtype=torch.int32)
    stop_ids = torch.zeros(max_reqs, w_stop, dtype=torch.int32)
    restore = torch.zeros(max_reqs, dtype=torch.int32)

    def _pick(pool, width):
        # Unique ids drawn from `pool`; a bias/stop id outside the allow-list would
        # already be -inf and make the feature invisible, so prefer the allow-list.
        n = int(torch.randint(1, min(width, pool.numel()) + 1, (1,), generator=gen))
        return pool[torch.randperm(pool.numel(), generator=gen)[:n]], n

    # Feature mix per request index (allowed, logit_bias, min_tokens/stop). Request
    # 0 uses everything so even a single-token batch hits all three paths.
    feature_mix = [
        (1, 1, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 0, 0),
    ]
    all_ids = torch.arange(vocab_size, dtype=torch.int32)
    for r in range(max_reqs):
        use_allowed, use_bias, use_stop = feature_mix[r % len(feature_mix)]
        allowed_pool = all_ids
        if use_allowed:
            ids, n = _pick(all_ids, w_allowed)
            num_allowed[r] = n
            allowed[r, :n] = ids
            allowed_pool = ids
        if use_bias:
            ids, n = _pick(allowed_pool, w_bias)
            num_bias[r] = n
            bias_ids[r, :n] = ids
            bias[r, :n] = torch.randn(n, generator=gen) * 5
        min_lens[r] = PROMPT_LEN + int(torch.randint(0, 12, (1,), generator=gen))
        if use_stop:
            ids, n = _pick(allowed_pool, w_stop)
            num_stop[r] = n
            stop_ids[r, :n] = ids
            if with_restore_flags:
                restore[r] = int(torch.rand(1, generator=gen) < 0.5)

    # Round-robin so every request type is hit whenever num_tokens >= max_reqs.
    mapping = (torch.arange(num_tokens) % max_reqs).to(torch.int32)
    pos = torch.randint(0, 14, (num_tokens,), generator=gen).to(torch.int32)

    state = dict(
        expanded_idx_mapping=mapping,
        pos=pos,
        num_allowed_token_ids=num_allowed,
        allowed_token_ids=allowed,
        num_logit_bias=num_bias,
        logit_bias_token_ids=bias_ids,
        logit_bias=bias,
        min_lens=min_lens,
        num_stop_token_ids=num_stop,
        restore_when_all_masked=restore,
        stop_token_ids=stop_ids,
    )
    return {k: v.to(flaggems_vllm.device) for k, v in state.items()}


def _run_gems(logits, state, check_all_masked_rows):
    with flaggems_vllm.use_gems():
        flaggems_vllm.apply_logit_bias(
            logits, check_all_masked_rows=check_all_masked_rows, **state
        )


def _run_vllm(logits, state, check_all_masked_rows):
    args = dict(state, logits=logits, check_all_masked_rows=check_all_masked_rows)
    vllm_apply_logit_bias(**{k: v for k, v in args.items() if k in _VLLM_PARAMS})


@pytest.mark.apply_logit_bias
@pytest.mark.skipif(vllm_apply_logit_bias is None, reason="need vLLM v1 GPU sampler")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("restore", [False, True])
def test_apply_logit_bias_vs_vllm(shape, widths, dtype, restore):
    if restore and not _VLLM_HAS_RESTORE:
        pytest.skip("installed vLLM has no restore_when_all_masked support")
    num_tokens, vocab_size = shape
    state = _make_state(
        num_tokens,
        vocab_size,
        widths,
        seed=num_tokens + vocab_size,
        with_restore_flags=restore,
    )
    logits = torch.randn(
        num_tokens, vocab_size, dtype=dtype, device=flaggems_vllm.device
    )
    if restore:
        # Leave only one stop token alive in every other row so the all-masked
        # branch is really exercised, not just the plain masking one.
        logits[::2] = -float("inf")
        stop = state["stop_token_ids"][state["expanded_idx_mapping"].long()][::2, 0]
        logits[torch.arange(0, num_tokens, 2, device=logits.device), stop.long()] = 1.0

    ref = logits.clone()
    _run_vllm(ref, state, restore)
    res = logits.clone()
    _run_gems(res, state, restore)

    utils.gems_assert_equal(res, ref)


@pytest.mark.apply_logit_bias
@pytest.mark.skipif(vllm_apply_logit_bias is None, reason="need vLLM v1 GPU sampler")
@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_logit_bias_row_stride(dtype):
    """logits may be a column slice of a wider buffer (stride(0) > vocab_size)."""
    num_tokens, vocab_size, pad = 6, 1000, 37
    state = _make_state(num_tokens, vocab_size, (16, 16, 8), seed=7)
    base = torch.randn(
        num_tokens, vocab_size + pad, dtype=dtype, device=flaggems_vllm.device
    )

    ref = base.clone()
    _run_vllm(ref[:, :vocab_size], state, False)
    res = base.clone()
    _run_gems(res[:, :vocab_size], state, False)

    # Compare the whole buffer so a write into the padding columns is also caught.
    utils.gems_assert_equal(res, ref)


def _single_row_state(structured):
    """One request per row using min_tokens with `STOP_TOKEN`; pos < min_len."""
    num_reqs = len(structured)
    device = flaggems_vllm.device
    zeros = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    return dict(
        expanded_idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        pos=torch.full((num_reqs,), 5, dtype=torch.int32, device=device),
        num_allowed_token_ids=zeros.clone(),
        allowed_token_ids=torch.zeros(num_reqs, 1, dtype=torch.int32, device=device),
        num_logit_bias=zeros.clone(),
        logit_bias_token_ids=torch.zeros(num_reqs, 1, dtype=torch.int32, device=device),
        logit_bias=torch.zeros(num_reqs, 1, dtype=torch.float32, device=device),
        # min_len = prompt_len + min_tokens, so pos=5 is below the minimum.
        min_lens=torch.full(
            (num_reqs,), PROMPT_LEN + MIN_TOKENS, dtype=torch.int32, device=device
        ),
        num_stop_token_ids=torch.ones(num_reqs, dtype=torch.int32, device=device),
        restore_when_all_masked=torch.tensor(
            structured, dtype=torch.int32, device=device
        ),
        stop_token_ids=torch.full(
            (num_reqs, 1), STOP_TOKEN, dtype=torch.int32, device=device
        ),
    )


def _only_stop_token_left(num_rows, vocab_size=128):
    """Logits as a terminal grammar leaves them: every token masked but one."""
    logits = torch.full(
        (num_rows, vocab_size), -float("inf"), device=flaggems_vllm.device
    )
    logits[:, STOP_TOKEN] = 1.0
    return logits


def _apply_restore(logits, structured):
    state = _single_row_state(structured)
    _run_gems(logits, state, check_all_masked_rows=any(structured))
    return logits.cpu()


# The four cases below are the assertions of vLLM's
# tests/v1/worker/test_gpu_logit_bias.py (commit 947dd62), run against the
# FlagGems kernel directly on tensors instead of through LogitBiasState.
@pytest.mark.apply_logit_bias
def test_min_tokens_restores_stop_token_when_row_would_be_empty():
    out = _apply_restore(_only_stop_token_left(1), [True])
    assert out[0, STOP_TOKEN] == 1.0


@pytest.mark.apply_logit_bias
def test_min_tokens_keeps_empty_row_without_structured_output():
    out = _apply_restore(_only_stop_token_left(1), [False])
    assert torch.isneginf(out).all()


@pytest.mark.apply_logit_bias
def test_min_tokens_keeps_masking_when_other_candidates_remain():
    logits = _only_stop_token_left(1)
    logits[0, OTHER_TOKEN] = 2.0
    out = _apply_restore(logits, [True])
    assert torch.isneginf(out[0, STOP_TOKEN])
    assert out[0, OTHER_TOKEN] == 2.0


@pytest.mark.apply_logit_bias
def test_min_tokens_mixed_batch_gates_restore_per_request():
    # CHECK_ALL_MASKED_ROWS is batch-wide, the per-request flag must still gate it.
    out = _apply_restore(_only_stop_token_left(2), [True, False])
    assert out[0, STOP_TOKEN] == 1.0
    assert torch.isneginf(out[1]).all()


@pytest.mark.apply_logit_bias
def test_min_tokens_does_not_restore_non_finite_stop_logit():
    # A stop token that was already -inf must not be "restored" to a legal token.
    logits = torch.full((1, 128), -float("inf"), device=flaggems_vllm.device)
    out = _apply_restore(logits, [True])
    assert torch.isneginf(out).all()


@pytest.mark.apply_logit_bias
@pytest.mark.parametrize("vocab_size", [5000, 70000])
def test_apply_logit_bias_bias_applied_exactly_once(vocab_size):
    # The kernel mutates logits in place and `+= bias` is not idempotent. The
    # first call per autotune key benchmarks every config on the same tensor, so
    # this guards against candidates leaking into the result (no vLLM needed).
    device = flaggems_vllm.device
    state = _single_row_state([0])
    state["num_stop_token_ids"].zero_()
    state["num_logit_bias"].fill_(2)
    state["logit_bias_token_ids"] = torch.tensor(
        [[7, 11]], dtype=torch.int32, device=device
    )
    state["logit_bias"] = torch.tensor(
        [[1.5, -2.0]], dtype=torch.float32, device=device
    )

    logits = torch.zeros(1, vocab_size, device=device)
    _run_gems(logits, state, False)

    expected = torch.zeros(1, vocab_size)
    expected[0, 7], expected[0, 11] = 1.5, -2.0
    utils.gems_assert_equal(logits.cpu(), expected)


@pytest.mark.apply_logit_bias
def test_apply_logit_bias_empty_tokens():
    logits = torch.empty(0, 128, device=flaggems_vllm.device)
    state = _make_state(0, 128, (4, 4, 4), seed=0)
    _run_gems(logits, state, False)
    assert logits.shape == (0, 128)


@pytest.mark.apply_logit_bias
def test_apply_logit_bias_unsupported_vocab_stride():
    logits = torch.randn(128, 4, device=flaggems_vllm.device).t()  # stride(1) == 4
    state = _make_state(4, 128, (4, 4, 4), seed=0)
    with pytest.raises(NotImplementedError):
        _run_gems(logits, state, False)
