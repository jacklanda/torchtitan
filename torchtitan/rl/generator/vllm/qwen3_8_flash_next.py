# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next generation with request-local lexical history."""

from dataclasses import dataclass, fields, replace

import spmd_types as spmd
import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.qwen3_8_flash_next.embedding import PLELayer
from torchtitan.models.qwen3_8_flash_next.row_sharded_embedding import (
    prepare_row_sharded_embedding,
)
from torchtitan.rl.generator.vllm.attention.qsa import PagedRingCache
from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper

# T = packed tokens, C = residual channels, D = model dimension,
# S = residual streams, N = hash heads, E = embedding dimension,
# P = pages, R = ring states, H = cache heads (one).


class VLLMPLELayer(PLELayer):
    @dataclass(kw_only=True, slots=True)
    class Config(PLELayer.Config):
        layer_idx: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.max_num_tokens = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
        )
        if (self.conv1d.kernel_size[0] - 1) * self.dilation >= 16:
            raise ValueError("Flash-Next PLE history must fit the 16-token state ring")
        self.cache = PagedRingCache(
            prefix=f"model.layers.{config.layer_idx}.ple.cache",
            num_heads=1,
            head_dim=config.dim * config.num_streams,
        )

    def forward(
        self,
        x_TC,
        *,
        input_ids_T,
        document_positions_T,
        routing=None,
        padding_mask_T=None,
    ):
        metadata = self.cache.metadata()
        embedding = self.ple_embedding
        if metadata is None or metadata.seq_lens.numel() == 0:
            hashes_TN = embedding.hash_ids(
                input_ids_T, document_positions_T=document_positions_T
            )
        else:
            request_T, positions_T, valid_T = metadata.token_coordinates(x_TC.shape[0])
            rows_T = torch.arange(x_TC.shape[0], device=x_TC.device)
            starts_T = metadata.query_start_loc[request_T]
            ids_cache_PRHC, _ = self.cache.views()
            shifted_T = [input_ids_T.long()]
            reset_T = torch.zeros_like(valid_T)
            for shift in range(1, embedding.ngram_size):
                previous_TC = self.cache.gather(
                    ids_cache_PRHC, request_T, positions_T - shift, metadata
                )[:, 0]
                previous_T = previous_TC[:, :4].contiguous().view(torch.int64)[:, 0]
                previous_T = torch.where(
                    rows_T - shift >= starts_T,
                    input_ids_T[(rows_T - shift).clamp_min(0)],
                    previous_T,
                )
                previous_T = torch.where(
                    (positions_T >= shift) & ~reset_T,
                    previous_T,
                    embedding.eos_token_id,
                )
                shifted_T.append(previous_T)
                reset_T = reset_T | (previous_T == embedding.eos_token_id)
            hashes = []
            mixed_T = shifted_T[0] * embedding.layer_multipliers[0]
            for shift in range(1, embedding.ngram_size):
                mixed_T = torch.bitwise_xor(
                    mixed_T, shifted_T[shift] * embedding.layer_multipliers[shift]
                )
                begin = (shift - 1) * embedding.heads_per_ngram
                end = begin + embedding.heads_per_ngram
                hashes.append(
                    mixed_T[:, None].remainder(
                        embedding.ngram_heads_vocab_sizes[begin:end]
                    )
                    + embedding.ngram_heads_offsets[begin:end]
                )
            hashes_TN = torch.cat(hashes, -1)
        table = embedding.ngram_embedding
        if table.row_mesh is not None:
            routing = prepare_row_sharded_embedding(
                hashes_TN,
                num_embeddings=table.num_embeddings,
                row_mesh=table.row_mesh,
                max_ids_per_owner=embedding.max_ids_per_owner(
                    self.max_num_tokens, num_owners=table.row_mesh.size()
                ),
            )
        embedding_TNE = table(hashes_TN, routing=routing, dtype=x_TC.dtype)
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            embedding_TNE = spmd.redistribute(
                embedding_TNE,
                tp_group,
                src=spmd.S(2),
                dst=spmd.R,
                backward_options={"op_dtype": embedding_TNE.dtype},
            )
        embedding_TE = embedding_TNE.flatten(-2)
        key_TSD = self.norm_key(self.key_proj(embedding_TE)).unflatten(
            -1, (self.num_streams, self.dim)
        )
        query_TSD = self.norm_query(x_TC).unflatten(-1, (self.num_streams, self.dim))
        score_TS = (key_TSD * query_TSD).sum(-1) * self.dim**-0.5
        score_TS = score_TS.abs().clamp_min(1e-6).sqrt() * score_TS.sign()
        value_TD = self.value_proj(embedding_TE)
        gated_TC = (score_TS.sigmoid()[:, :, None] * value_TD[:, None]).flatten(-2)
        normed_TC = self.norm_conv(gated_TC)
        if metadata is None or metadata.seq_lens.numel() == 0:
            return torch.zeros_like(x_TC)
        normed_TC = normed_TC.masked_fill(~valid_T[:, None], 0)
        _, normed_cache_PRHC = self.cache.views()
        conv_TC = torch.zeros_like(normed_TC, dtype=torch.float32)
        for tap in range(self.conv1d.kernel_size[0]):
            shift = (self.conv1d.kernel_size[0] - 1 - tap) * self.dilation
            source_T = positions_T - shift
            shifted_TC = torch.where(
                (rows_T - shift >= starts_T)[:, None],
                normed_TC[(rows_T - shift).clamp_min(0)],
                self.cache.gather(normed_cache_PRHC, request_T, source_T, metadata)[
                    :, 0
                ],
            )
            shifted_TC = shifted_TC.float().masked_fill((source_T < 0)[:, None], 0)
            conv_TC = conv_TC + shifted_TC * self.conv1d.weight[:, 0, tap].float()
        # Write only each request's last 16 new tokens, once its old history is
        # no longer needed. Those ring destinations are unique even for prefill.
        ring_capacity = normed_cache_PRHC.shape[1]
        slots_T = (
            metadata.block_table[request_T, 0].long() * ring_capacity
            + positions_T % ring_capacity
        )
        slots_T = slots_T.masked_fill(
            ~valid_T | (positions_T < metadata.seq_lens[request_T] - 16), -1
        )
        ids_TC = torch.zeros_like(x_TC)
        ids_TC[:, :4] = input_ids_T.long().contiguous().view(x_TC.dtype).reshape(-1, 4)
        self.cache.write(
            ids_TC[:, None], normed_TC[:, None], replace(metadata, slot_mapping=slots_T)
        )
        return (gated_TC + F.silu(conv_TC.to(normed_TC.dtype))).masked_fill(
            ~valid_T[:, None], 0
        )


class VLLMFlashNextModelWrapper(VLLMModelWrapper):
    def __init__(self, *, model_config, **kwargs):
        if kwargs["vllm_config"].cache_config.enable_prefix_caching:
            raise ValueError(
                "Flash-Next PLE request history requires enable_prefix_caching=False"
            )
        layers = []
        for layer_idx, layer in enumerate(model_config.layers):
            if layer.ple is not None:
                ple = VLLMPLELayer.Config(
                    **{
                        field.name: getattr(layer.ple, field.name)
                        for field in fields(layer.ple)
                    },
                    layer_idx=layer_idx,
                )
                layer = replace(layer, ple=ple)
            layers.append(layer)
        super().__init__(model_config=replace(model_config, layers=layers), **kwargs)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        if input_ids is None or positions is None or inputs_embeds is not None:
            raise ValueError(
                "Flash-Next generation requires token IDs and text positions"
            )
        with self.parallelism_context.activate_spmd():
            return self.model(
                input_ids,
                positions=positions,
                document_positions=positions,
                ple_input_ids=input_ids,
                attention_metadata=None,
                **self._prepare_step_kwargs(input_ids.shape[0]),
            )
