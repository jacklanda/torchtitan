# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Kimi Delta Attention recurrence over a paged state pool, shared by the engines'
KDA adapters: the trainer's Attention Gym kernels, run on the engine's state slots.

Legend: B = rows, T = tokens per row, C = rank-local q|k|v channels, H = heads,
K = key head dim, V = value head dim (V == K), N = state slots.
"""

import torch

from torchtitan.models.common.attention.linear_attention import chunk_kernel_options
from torchtitan.models.common.attn_gym import bound_gate, l2norm, paged_chunk_kda
from torchtitan.rl.generator.attention.state_checkpoints import (
    copy_checkpoints,
    periodic_checkpoints,
)

# Attention Gym's replay state: the q, k, v, gate and beta of each slot's unfinished
# chunk, and its token count.
# TODO(https://github.com/meta-pytorch/attention-gym/issues/692): Attention Gym
# documents this layout (ReplayState) but exposes no constants or constructor for it;
# take them from Attention Gym.
REPLAY_CHUNK_SIZE = 64
REPLAY_STATE_DTYPES = (
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
    torch.float32,
    torch.int32,
)


def replay_state_shapes(
    local_num_heads: int, head_dim: int
) -> tuple[tuple[int, ...], ...]:
    """Return the per-slot shapes of ``REPLAY_STATE_DTYPES``'s tensors."""
    return (
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads),
        (1,),
    )


def paged_kda(
    convolved_BTC: torch.Tensor,
    raw_gate_BTHK: torch.Tensor,
    raw_beta_BTH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    state_pool_NHVK: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    replay_state: tuple[torch.Tensor, ...] | None,
    head_dim: int,
    lower_bound: float,
    checkpoint_slots: torch.Tensor | None = None,
    checkpoint_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the recurrence of each sequence from its slot in ``state_indices``,
    advancing ``state_pool_NHVK`` (and ``replay_state``) in place.

    Packed prefill passes ``B == 1`` with ``cu_seqlens``; decode passes one token per
    row and no ``cu_seqlens``. With ``replay_state`` (batch-invariant mode), the slot's
    state is the one after its last complete chunk plus the replayed unfinished chunk,
    which matches the trainer's chunk kernel bitwise.

    Args:
        convolved_BTC: The post-convolution ``q | k | v`` channels, rank-local.
        A_log_H, dt_bias_HK: The gate parameters, in float32.
        has_initial_state: Per sequence, whether its slot holds a state to resume;
            None resumes every slot.
        checkpoint_slots, checkpoint_lens: Packed prefill only. Per sequence, a slot
            that also receives the state after the sequence's first ``checkpoint_lens``
            tokens (non-positive slot: none; lengths a positive multiple of 64), for an
            engine's prefix cache. With ``replay_state``, the slot also receives the
            snapshot's replay state.

    Returns:
        ``(B, T, H, K)``.
    """
    batch_size, num_tokens = convolved_BTC.shape[:2]
    query_BTHK, key_BTHK, value_BTHK = (
        tensor.reshape(batch_size, num_tokens, -1, head_dim)
        for tensor in convolved_BTC.chunk(3, dim=-1)
    )
    inputs = (
        l2norm(query_BTHK),
        l2norm(key_BTHK),
        value_BTHK,
        bound_gate(raw_gate_BTHK, A_log_H, dt_bias_HK, lower_bound=lower_bound),
        raw_beta_BTH.float().sigmoid(),
    )
    checkpoints = replayed_S = None
    if checkpoint_slots is not None:
        assert cu_seqlens is not None and has_initial_state is not None
        if replay_state is not None:
            # A replay checkpoint interval counts from the start of the slot's
            # unfinished chunk: its replayed tokens come first.
            count_N = replay_state[-1][:, 0]
            replayed_S = torch.where(
                has_initial_state, count_N[state_indices.clamp(min=0).long()], 0
            )
        checkpoints = periodic_checkpoints(
            cu_seqlens, state_pool_NHVK, num_tokens=num_tokens, replayed_S=replayed_S
        )
    output = paged_chunk_kda(
        *inputs,
        state_pool_NHVK,
        state_indices,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial_state,
        replay_state=replay_state,
        kernel_options=(
            None if cu_seqlens is None else chunk_kernel_options(query_BTHK)
        ),
        state_checkpoints=checkpoints,
    )
    if checkpoints is not None:
        copy_checkpoints(
            checkpoints,
            state_pool_NHVK,
            checkpoint_slots_S=checkpoint_slots,
            checkpoint_lens_S=checkpoint_lens,
            replayed_S=replayed_S,
        )
        if replay_state is not None:
            _copy_replay_tails(
                inputs,
                replay_state,
                cu_seqlens=cu_seqlens,
                checkpoint_slots_S=checkpoint_slots,
                checkpoint_lens_S=checkpoint_lens,
                replayed_S=replayed_S,
            )
    return output


def _copy_replay_tails(
    inputs: tuple[torch.Tensor, ...],
    replay_state: tuple[torch.Tensor, ...],
    *,
    cu_seqlens: torch.Tensor,
    checkpoint_slots_S: torch.Tensor,
    checkpoint_lens_S: torch.Tensor,
    replayed_S: torch.Tensor,
) -> None:
    """Give each checkpoint slot its snapshot's replay state: the tokens after the
    snapshot's last complete chunk, and their count.

    A snapshot lies at least one chunk past its sequence's start, so its last complete
    chunk ends after the replayed tokens, and those tokens are this call's inputs.
    """
    selected_S = checkpoint_slots_S > 0
    tail_S = (replayed_S + checkpoint_lens_S) % REPLAY_CHUNK_SIZE
    # The packed row of each snapshot's first replay token.
    first_S = cu_seqlens[:-1] + checkpoint_lens_S - tail_S
    token = torch.arange(REPLAY_CHUNK_SIZE, device=first_S.device)
    num_tokens = inputs[0].shape[1]
    rows_SR = (first_S[:, None] + token).clamp(0, num_tokens - 1)
    stored_SR = selected_S[:, None] & (token < tail_S[:, None])
    slots_S = torch.where(selected_S, checkpoint_slots_S, 0).long()
    # Unselected sequences write slot 0 back to itself.
    for source, cache in zip(inputs, replay_state[:-1], strict=True):
        tokens = source.reshape(num_tokens, *source.shape[2:])[rows_SR]
        trailing = (1,) * (tokens.ndim - 2)
        cache[slots_S] = torch.where(
            stored_SR.view(*stored_SR.shape, *trailing),
            tokens.to(cache.dtype),
            cache[slots_S],
        )
    count_N = replay_state[-1][:, 0]
    count_N[slots_S] = torch.where(
        selected_S, tail_S.to(count_N.dtype), count_N[slots_S]
    )
