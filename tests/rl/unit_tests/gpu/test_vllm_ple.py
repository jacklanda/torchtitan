# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "attn_gym.sparse.paged_decode",
    reason="Flash-Next cache tests require Attention Gym paged sparse attention",
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.rl.generator.vllm import qwen3_8_flash_next
from torchtitan.rl.generator.vllm.attention import qsa

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PLE cache requires a GPU"
)


def test_ple_chunked_history_reordering_eos_and_page_reuse(monkeypatch):
    """Split PLE execution must match full requests even after reordering and slot reuse."""
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
    )
    monkeypatch.setattr(qsa, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(qwen3_8_flash_next, "get_current_vllm_config", lambda: config)
    context = {}
    monkeypatch.setattr(
        qsa, "get_forward_context", lambda: SimpleNamespace(attn_metadata=context)
    )
    torch.manual_seed(42)
    ple_config = (
        build_model_config("debugmodel", seq_len=64, parallelism=ParallelismConfig())
        .layers[1]
        .ple
    )
    ple_config.ple_embedding.eos_token_id = 63
    reference = ple_config.build().cuda()
    reference.init_states()
    for parameter in reference.parameters():
        parameter.data = parameter.data.bfloat16()
    cached = (
        qwen3_8_flash_next.VLLMPLELayer.Config(
            **{
                field.name: getattr(ple_config, field.name)
                for field in fields(ple_config)
            },
            layer_idx=1,
        )
        .build()
        .cuda()
    )
    cached.init_states()
    for parameter in cached.parameters():
        parameter.data = parameter.data.bfloat16()
    cached.load_state_dict(reference.state_dict(), strict=True)
    reference.requires_grad_(False)
    cached.requires_grad_(False)
    channels = ple_config.dim * ple_config.num_streams
    cached.cache.bind_kv_cache(
        torch.randn(6, 1, 32, 2 * channels, device="cuda", dtype=torch.bfloat16)
    )
    ids = [torch.randint(0, 248320, (27,), device="cuda") for _ in range(3)]
    ids[0][1] = 0x7FC1
    ids[0][2] = ids[0][15] = 63
    inputs = [
        torch.randn(27, channels, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    expected = [
        reference(
            x, input_ids_T=tokens, document_positions_T=torch.arange(27, device="cuda")
        )
        for x, tokens in zip(inputs, ids)
    ]
    completed = [0, 0, 0]
    # Request 2 reuses request 0's pages after it completes.
    pages = [[4, 1], [3, 5], [4, 1]]
    for step in (
        [(0, 3), (1, 11)],
        [(1, 1), (0, 14)],
        [(0, 10), (1, 15)],
        [(2, 1)],
        [(2, 26)],
    ):
        offsets = [0]
        positions, tokens, x, slots, lengths, table = [], [], [], [], [], []
        for request, count in step:
            start = completed[request]
            position = torch.arange(start, start + count, device="cuda")
            positions.append(position)
            tokens.append(ids[request][start : start + count])
            x.append(inputs[request][start : start + count])
            table.append(pages[request])
            slots.append(
                torch.tensor(pages[request], device="cuda")[position // 16] * 16
                + position % 16
            )
            lengths.append(start + count)
            offsets.append(offsets[-1] + count)
        context[cached.cache.prefix] = qsa.QSAPagedMetadata(
            block_table=torch.tensor(table, device="cuda", dtype=torch.int32),
            slot_mapping=torch.cat(slots).long(),
            query_start_loc=torch.tensor(offsets, device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            max_query_len=max(count for _, count in step),
        )
        actual = cached(
            torch.cat(x),
            input_ids_T=torch.cat(tokens),
            document_positions_T=torch.cat(positions),
        )
        for row, (request, count) in enumerate(step):
            start = completed[request]
            torch.testing.assert_close(
                actual[offsets[row] : offsets[row + 1]],
                expected[request][start : start + count],
                atol=0,
                rtol=0,
            )
            completed[request] += count
