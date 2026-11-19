# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cache addressing, causal tails and request isolation for Flash-Next generation."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "attn_gym.sparse.indexer.paged",
    reason="Paged QSA tests require the Attention Gym decode indexer",
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.rl.generator.vllm.attention import qsa
from torchtitan.rl.generator.vllm.attention.qsa_model import VLLMQSAIndexer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Paged QSA requires a GPU"
)


@pytest.fixture
def make_cache(monkeypatch):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(qsa, "get_current_vllm_config", lambda: config)
    metadata = {}
    monkeypatch.setattr(
        qsa, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )

    def build(prefix, head_dim, num_heads=1, num_pages=8):
        cache = qsa.PagedCache(prefix=prefix, head_dim=head_dim, num_heads=num_heads)
        cache.bind_kv_cache(
            torch.randn(
                num_pages,
                num_heads,
                16,
                2 * head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )
        return cache, metadata

    return build


def test_paged_qsa_mixed_queries_tail_and_padding(make_cache):
    """A reordered physical cache must match sparse causal attention, including tails."""
    torch.manual_seed(42)
    cache, context = make_cache("qsa", 32, num_heads=2)
    lengths, query_lengths = [19, 7, 35], [3, 7, 1]
    block_table = torch.tensor(
        [[5, 1, 0], [3, 0, 0], [6, 2, 4]], device="cuda", dtype=torch.int32
    )
    offsets = torch.tensor([0, 3, 10, 11], device="cuda", dtype=torch.int32)
    slots = []
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        for position in range(length - query_length, length):
            slots.append(block_table[request, position // 16] * 16 + position % 16)
    metadata = qsa.QSAPagedMetadata(
        block_table=block_table,
        slot_mapping=torch.stack(slots).long(),
        query_start_loc=offsets,
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_query_len=7,
    )
    context["qsa"] = metadata
    query = torch.randn(13, 4, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(13, 2, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Select just the first complete block. Later tokens use it plus their tail.
    blocks = torch.full((13, 1), -1, device="cuda", dtype=torch.int32)
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        start = int(offsets[request])
        for row in range(query_length):
            if length - query_length + row + 1 >= 4:
                blocks[start + row, 0] = 0
    output = cache.attend(query, key, value, blocks, scale=32**-0.5)
    key_cache, value_cache = cache.views()
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        for row in range(query_length):
            index = int(offsets[request]) + row
            count = length - query_length + row + 1
            selected = list(range(4)) if count >= 4 else []
            selected += list(range(count // 4 * 4, count))
            selected = torch.tensor(selected, device="cuda")
            pages = block_table[request, selected // 16].long()
            keys = key_cache[pages, selected % 16].repeat_interleave(2, 1).float()
            values = value_cache[pages, selected % 16].repeat_interleave(2, 1).float()
            logits = torch.einsum("hd,shd->hs", query[index].float(), keys) * 32**-0.5
            expected = torch.einsum("hs,shd->hd", logits.softmax(-1), values)
            torch.testing.assert_close(
                output[index].float(), expected, atol=1e-2, rtol=1e-2
            )
    assert torch.count_nonzero(output[11:]) == 0


def _reference_paged_indexer(
    query,
    key_cache,
    weights,
    block_table,
    cu_seqlens_q,
    seqused_k,
    *,
    topk,
    compress_ratio=4,
):
    """Test-only implementation of the agreed Attention Gym selection contract."""
    result = torch.full(
        (query.shape[0], topk), -1, device=query.device, dtype=torch.int32
    )
    for request in range(seqused_k.numel()):
        start, end = map(int, cu_seqlens_q[request : request + 2])
        for row in range(start, end):
            count = (
                int(seqused_k[request]) - (end - start) + row - start + 1
            ) // compress_ratio
            blocks = torch.arange(count, device=query.device)
            keys = key_cache[
                block_table[request, blocks // key_cache.shape[1]].long(),
                blocks % key_cache.shape[1],
            ]
            scores = (
                torch.einsum("hd,bd->hb", query[row].float(), keys.float()).relu()
                * weights[row, :, None]
            ).sum(0)
            selected = scores.argsort(descending=True, stable=True)[:topk].sort().values
            result[row, : selected.numel()] = selected.int()
    return result


def test_indexer_completed_blocks_across_prefill_and_decode(make_cache):
    """Pooling straddling chunks must equal norm(mean(raw keys)) then block-start RoPE."""
    make_cache("fixture", 32)
    torch.manual_seed(42)
    config = (
        build_model_config("debugmodel", seq_len=64, parallelism=ParallelismConfig())
        .layers[3]
        .attention.indexer
    )
    config.token_budget = 8
    indexer = (
        VLLMQSAIndexer.Config(
            **{field.name: getattr(config, field.name) for field in fields(config)},
            layer_idx=3,
        )
        .build()
        .cuda()
    )
    indexer.init_states()
    for parameter in indexer.parameters():
        parameter.data = parameter.data.bfloat16()
    indexer.requires_grad_(False)
    indexer.cache.bind_kv_cache(
        torch.zeros(3, 1, 4, 2 * config.head_dim, device="cuda", dtype=torch.bfloat16)
    )
    indexer.raw_cache.bind_kv_cache(
        torch.randn(2, 1, 32, 2 * config.head_dim, device="cuda", dtype=torch.bfloat16)
    )
    context = qsa.get_forward_context().attn_metadata
    raw_keys = []
    offset = 0
    block_table = torch.tensor([[2, 1]], device="cuda", dtype=torch.int32)
    for length in (3, 1, 9, 6, 1):
        positions = torch.arange(offset, offset + length, device="cuda")
        x = torch.randn(
            length,
            config.index_qk_proj.in_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        raw_keys.append(indexer.index_qk_proj(x)[:, -config.head_dim :])
        metadata = qsa.QSAPagedMetadata(
            block_table=block_table,
            slot_mapping=(block_table[0, positions // 16].long() * 16 + positions % 16),
            query_start_loc=torch.tensor([0, length], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor([offset + length], device="cuda", dtype=torch.int32),
            max_query_len=length,
        )
        context[indexer.cache.prefix] = metadata
        context[indexer.raw_cache.prefix] = replace(
            metadata, block_table=torch.ones(1, 1, device="cuda", dtype=torch.int32)
        )
        selected, _ = indexer(x, positions=positions, document_positions_T=positions)
        offset += length
        count = offset // 4
        if count:
            pooled = (
                torch.cat(raw_keys)[: count * 4]
                .reshape(count, 4, -1)
                .float()
                .mean(1)
                .bfloat16()
            )
            expected = indexer._rotate(
                indexer.k_layernorm(pooled)[:, None],
                torch.arange(count, device="cuda") * 4,
            )[:, 0]
            cache, _ = indexer.cache.views()
            blocks = torch.arange(count, device="cuda")
            actual = cache[block_table[0, blocks // 4].long(), blocks % 4, 0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        query = indexer._rotate(
            indexer.q_layernorm(
                indexer.index_qk_proj(x)[:, : -config.head_dim].unflatten(
                    -1, (indexer.num_heads, config.head_dim)
                )
            ),
            positions,
        )
        expected_selection = _reference_paged_indexer(
            query,
            indexer.cache.views()[0][:, :, 0],
            query.new_ones(query.shape[:2]),
            block_table,
            metadata.query_start_loc,
            metadata.seq_lens,
            topk=indexer.block_topk,
        )
        torch.testing.assert_close(selected, expected_selection, atol=0, rtol=0)
        for row, position in enumerate(positions.tolist()):
            assert ((selected[row] < (position + 1) // 4) | (selected[row] == -1)).all()


def test_decode_graph_refreshes_request_layout(make_cache):
    """Replay must consume new block tables, lengths and padding."""
    cache, context = make_cache("graph", 32)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=64),
    )
    builder = qsa.QSAPagedMetadataBuilder(
        SimpleNamespace(
            block_size=16,
            uses_slot_mapping=True,
            max_num_blocks_per_req=lambda config, max_len: 4,
        ),
        ["graph"],
        config,
        torch.device("cuda"),
    )

    def metadata(pages, lengths):
        count = len(pages)
        block_table = torch.zeros(count, 8, device="cuda", dtype=torch.int32)
        block_table[:, 0] = torch.tensor(pages, device="cuda", dtype=torch.int32)
        return builder.build(
            0,
            SimpleNamespace(
                num_reqs=count,
                num_actual_tokens=4,
                max_query_len=1,
                block_table_tensor=block_table,
                query_start_loc=torch.arange(
                    count + 1, device="cuda", dtype=torch.int32
                ),
                seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                slot_mapping=torch.tensor(
                    [page * 16 + length - 1 for page, length in zip(pages, lengths)],
                    device="cuda",
                    dtype=torch.int64,
                ),
            ),
        )

    query = torch.randn(4, 2, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(4, 1, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    blocks = torch.zeros(4, 1, device="cuda", dtype=torch.int32)
    context["graph"] = metadata([1, 2, 3, 4], [8, 9, 10, 11])

    def step():
        return cache.attend(query, key, value, blocks, scale=32**-0.5)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = step()
    context["graph"] = metadata([6, 1], [13, 7])
    expected = step()
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[2:]) == 0
