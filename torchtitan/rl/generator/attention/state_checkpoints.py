# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prefill snapshots of paged linear attention states, for an engine's prefix cache:
each selected sequence's state after a given number of its tokens, copied into
another state slot, from Attention Gym's periodic state checkpoints.

Legend: T = tokens of the step, N = state slots, H = value heads, K = key head dim,
V = value head dim, S = sequences.
"""

import torch
import torch.nn.functional as F

from torchtitan.models.common.attn_gym import StateCheckpoints

# Attention Gym's chunk size, the finest checkpoint interval it supports.
CHECKPOINT_INTERVAL = 64


def periodic_checkpoints(
    cu_seqlens: torch.Tensor,
    state_pool_NHVK: torch.Tensor,
    *,
    num_tokens: int,
    replayed_S: torch.Tensor | None = None,
) -> StateCheckpoints:
    """Allocate Attention Gym checkpoints of every sequence's state every
    ``CHECKPOINT_INTERVAL`` tokens; ``replayed_S`` are the tokens a KDA replay state
    adds before each sequence's tokens, from which its intervals count."""
    lens_S = cu_seqlens.diff()
    if replayed_S is not None:
        lens_S = lens_S + replayed_S
    counts_S = lens_S.div(CHECKPOINT_INTERVAL, rounding_mode="floor")
    num_seqs = cu_seqlens.shape[0] - 1
    # sum((replayed + len) // 64) <= (T + 63 S) // 64, from shapes only.
    capacity = num_tokens // CHECKPOINT_INTERVAL + num_seqs
    return StateCheckpoints(
        state_pool_NHVK.new_empty(
            (capacity, *state_pool_NHVK.shape[1:]), dtype=torch.float32
        ),
        F.pad(counts_S.cumsum(0), (1, 0)).to(torch.int32),
        CHECKPOINT_INTERVAL,
    )


def copy_checkpoints(
    checkpoints: StateCheckpoints,
    state_pool_NHVK: torch.Tensor,
    *,
    checkpoint_slots_S: torch.Tensor,
    checkpoint_lens_S: torch.Tensor,
    replayed_S: torch.Tensor | None = None,
) -> None:
    """Copy each sequence's checkpoint at its last interval boundary within its first
    ``checkpoint_lens_S`` tokens into its ``checkpoint_slots_S`` slot (non-positive:
    none). Lengths of selected sequences are at least ``CHECKPOINT_INTERVAL``."""
    states_NHVK, offsets_S, interval = checkpoints
    total_S = checkpoint_lens_S
    if replayed_S is not None:
        total_S = total_S + replayed_S
    selected_S = checkpoint_slots_S > 0
    rows_S = offsets_S[:-1] + total_S.div(interval, rounding_mode="floor") - 1
    rows_S = torch.where(selected_S, rows_S, 0).long()
    slots_S = torch.where(selected_S, checkpoint_slots_S, 0).long()
    # Unselected sequences write slot 0 back to itself.
    state_pool_NHVK[slots_S] = torch.where(
        selected_S.view(-1, 1, 1, 1),
        states_NHVK[rows_S].to(state_pool_NHVK.dtype),
        state_pool_NHVK[slots_S],
    )
