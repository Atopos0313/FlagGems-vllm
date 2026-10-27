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

"""Fused logit-bias kernel for the vLLM v1 GPU model runner sampler.

Aligned with ``vllm/v1/worker/gpu/sample/logit_bias.py::apply_logit_bias``
(vLLM commit 947dd62). For every token row (indexed through
``expanded_idx_mapping`` into per-request state) it applies, in place:

1. ``allowed_token_ids``: every logit outside the allow-list becomes ``-inf``.
2. ``logit_bias``: ``logits[token_id] += bias`` for the listed tokens.
3. ``min_tokens``: while ``pos + 1 < min_len`` the stop tokens become ``-inf``.
   If ``check_all_masked_rows`` is set and the request opted in through
   ``restore_when_all_masked``, the stop tokens are restored when masking them
   would leave the whole row at ``-inf`` (structured output reached a terminal
   state).
"""

import logging

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# `logits` is rewritten in place and `logits[token] += bias` is not idempotent,
# so every autotune candidate must start from the original logits.
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("apply_logit_bias"),
    key=["vocab_size"],
    strategy=["log"],
    restore_value=["logits_ptr"],
)
@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore_stop_logits = tl.load(
                restore_when_all_masked_ptr + req_state_idx
            )
            if should_restore_stop_logits:
                stop_logits = tl.load(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    mask=mask,
                    other=-float("inf"),
                )

                # Save must read original logits before the -inf overwrite.
                tl.debug_barrier()

                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )

                # The -inf overwrite must be visible to the row scan below.
                tl.debug_barrier()

                row_max = tl.full((), -float("inf"), tl.float32)
                for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
                    row = tl.load(
                        logits_ptr + token_idx * logits_stride + offset,
                        mask=offset < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(row, axis=0))

                if row_max == -float("inf"):
                    tl.store(
                        logits_ptr + token_idx * logits_stride + stop_token_ids,
                        stop_logits,
                        mask=mask
                        & (stop_logits > -float("inf"))
                        & (stop_logits < float("inf")),
                    )
            else:
                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )
        else:
            tl.store(
                logits_ptr + token_idx * logits_stride + stop_token_ids,
                -float("inf"),
                mask=mask,
            )


def apply_logit_bias(
    logits,
    expanded_idx_mapping,
    pos,
    num_allowed_token_ids,
    allowed_token_ids,
    num_logit_bias,
    logit_bias_token_ids,
    logit_bias,
    min_lens,
    num_stop_token_ids,
    restore_when_all_masked,
    stop_token_ids,
    check_all_masked_rows=False,
):
    """Apply allowed-token mask, logit bias and min-tokens masking in place.

    Args:
        logits: ``[num_tokens, vocab_size]``, last dim must be contiguous. Modified
            in place.
        expanded_idx_mapping: ``[num_tokens]`` int32, token row -> request state row.
        pos: ``[num_tokens]`` int32, position of each token row.
        num_allowed_token_ids: ``[max_num_reqs]`` int32 count per request.
        allowed_token_ids: ``[max_num_reqs, W_allowed]`` int32.
        num_logit_bias: ``[max_num_reqs]`` int32 count per request.
        logit_bias_token_ids: ``[max_num_reqs, W_bias]`` int32, unique per request.
        logit_bias: ``[max_num_reqs, W_bias]`` float bias values.
        min_lens: ``[max_num_reqs]`` int32, ``prompt_len + min_tokens``.
        num_stop_token_ids: ``[max_num_reqs]`` int32 count per request.
        restore_when_all_masked: ``[max_num_reqs]`` int32 flag per request.
        stop_token_ids: ``[max_num_reqs, W_stop]`` int32, unique per request.
        check_all_masked_rows: enable the "restore stop tokens when the row would be
            all ``-inf``" path; only requests with ``restore_when_all_masked`` set
            are affected.

    Returns:
        None. ``logits`` is updated in place (same contract as vLLM).
    """
    logger.debug("GEMS APPLY LOGIT BIAS")
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {logits.dim()}D")
    num_tokens, vocab_size = logits.shape
    if expanded_idx_mapping.numel() != num_tokens or pos.numel() != num_tokens:
        raise ValueError(
            "expanded_idx_mapping and pos must have one entry per logits row"
        )
    # The kernel indexes rows as `row * stride(0) + token_id`, so the vocab dim
    # of logits and the last dim of every per-request table must be unit-stride.
    for name, tensor in (
        ("logits", logits),
        ("allowed_token_ids", allowed_token_ids),
        ("logit_bias_token_ids", logit_bias_token_ids),
        ("logit_bias", logit_bias),
        ("stop_token_ids", stop_token_ids),
    ):
        if tensor.dim() != 2:
            raise ValueError(f"{name} must be 2D, got {tensor.dim()}D")
        if tensor.stride(1) != 1:
            raise NotImplementedError(
                f"apply_logit_bias requires {name} to be contiguous on its last dim"
            )

    if num_tokens == 0 or vocab_size == 0:
        return

    # BLOCK_SIZE has to cover the widest per-request table (it is a correctness
    # bound, not a tuning knob); the tunable knobs are LOGITS_BLOCK_SIZE/num_warps.
    block_size = triton.next_power_of_2(
        max(
            allowed_token_ids.shape[-1],
            logit_bias_token_ids.shape[-1],
            stop_token_ids.shape[-1],
            1,
        )
    )
    _bias_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        restore_when_all_masked,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=block_size,
        CHECK_ALL_MASKED_ROWS=bool(check_all_masked_rows),
    )
