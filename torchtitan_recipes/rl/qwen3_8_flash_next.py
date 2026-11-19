# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Disaggregated Flash-Next GRPO with GPU-resident frozen lexical embeddings."""

from dataclasses import dataclass, fields

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.embedding import NGramTable, PLELayer
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.dapo_math.data import AIME2025Dataset, DapoMathDataset
from torchtitan.rl.examples.dapo_math.env import DapoMathEnv
from torchtitan.rl.examples.dapo_math.rubric import RewardMathVerify
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import MessageEnvRolloutWorker, Rollouter
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses import DAPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from torchtitan_recipes.rl.model_configs import build_rl_model_configs


class _FrozenNGramTable(NGramTable):
    @dataclass(kw_only=True, slots=True)
    class Config(NGramTable.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.weight.requires_grad_(False)


def qwen3_8_flash_next_dapo_disaggregated(
    *, hf_assets_path: str, dapo_math_repo_id: str
) -> Controller.Config:
    """Thirty steps, 128 samples/step, 32 trainer GPUs and two 8-GPU generators.

    PLE and the indexer are frozen and loaded from the same HF checkpoint in both
    roles. PLE shards stay on GPU. The generators reuse model weights in place,
    and generated log-probabilities supply GRPO's old policy.
    """
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        pipeline_parallel_degree=8,
        expert_parallel_degree=4,
        enable_sequence_parallel=True,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=4,
        tensor_parallel_degree=2,
        expert_parallel_degree=8,
        enable_sequence_parallel=True,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "125B-A6B-text",
        seq_len=8192,
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    for model in (trainer_model, generator_model):
        for _, ple, _, _ in model.traverse(PLELayer.Config):
            table = ple.ple_embedding.ngram_embedding
            ple.ple_embedding.ngram_embedding = _FrozenNGramTable.Config(
                **{field.name: getattr(table, field.name) for field in fields(table)}
            )
    config = Controller.Config(
        hf_assets_path=hf_assets_path,
        dump_folder="outputs/rl/qwen3_8_flash_next_dapo",
        renderer=from_renderers(Qwen35RendererConfig(enable_thinking=True)),
        precision=PrecisionConfig(dtype="float32", param_dtype="bfloat16"),
        async_loop=AsyncLoopConfig(
            num_training_steps=30,
            num_prompts_per_train_step=16,
            num_samples_per_prompt=8,
            target_offpolicy_steps=1,
            validation=ValidationConfig(steps=0),
            batcher=Batcher.Config(
                training_sample_builder=TrainingSampleBuilder.Config(
                    drop_zero_std_reward_groups=False
                )
            ),
        ),
        rollouter=Rollouter.Config(
            training_dataloader=IterableRolloutDataLoader.Config(
                dataset=DapoMathDataset.Config(repo_id=dapo_math_repo_id)
            ),
            validation_dataset=AIME2025Dataset.Config(),
            worker=MessageEnvRolloutWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardMathVerify.Config(weight=1.0)], error_reward=0.0
                ),
                message_env=DapoMathEnv.Config(),
                token_env=TokenEnv.Config(max_rollout_tokens=8192, max_num_turns=1),
                advantage=AdvantageEstimator.Config(should_std_normalize=True),
            ),
        ),
        trainer=Trainer.Config(
            model=trainer_model,
            parallelism=trainer_parallelism,
            training=TrainingConfig(
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=8192,
            ),
            activation_remat=ActivationRematConfig(save_regions=[]),
            dist_moe_runtime=DistMoeRuntime.Config(scratch_capacity_factor=4.0),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True, initial_load_path=hf_assets_path
            ),
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*", lr=1e-6, betas=(0.9, 0.98), weight_decay=0.1
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0, min_lr_factor=1.0
                ),
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=trainer_model.vocab_size,
                ),
            ),
        ),
        num_generators=2,
        generator=Generator.Config(
            model=generator_model,
            parallelism=generator_parallelism,
            engine=VLLMEngine.Config(
                cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY"),
                extra_engine_args={"enable_prefix_caching": False},
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True, initial_load_path=hf_assets_path
            ),
            gpu_memory_limit=0.85,
            max_num_batched_tokens=8192,
            dist_moe_runtime=DistMoeRuntime.Config(
                inference=True, scratch_capacity_factor=8.0
            ),
            sampling=SamplingConfig(temperature=0.8, top_p=1.0, max_tokens=4096),
        ),
    )
    config.trainer = apply_transforms(
        config.trainer, [DistMoeTransform(expert_precision="bf16")]
    )
    config.generator = apply_transforms(
        config.generator, [DistMoeTransform(expert_precision="bf16")]
    )
    return config
