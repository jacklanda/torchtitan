# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cache addressing, causal tails and request isolation for Flash-Next generation."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "attn_gym.sparse.paged_decode",
    reason="Paged QSA tests require Attention Gym paged sparse attention",
)

from torchtitan.rl.generator.vllm.attention import qsa

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
