#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import sys
import argparse
import logging
import math
import os
from pathlib import Path
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from torchvision.utils import save_image
import diffusers
from diffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn as nn
import torch.nn.functional as F
import warnings

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.rellisur_language_dataset import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_SCHEMA,
    RELLISURLanguageDataset,
)
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
import copy

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))




class LanguageGuidedSigmaPredictor(nn.Module):
    """Predicts spatial sigma with FiLM modulation from degradation language attributes."""

    def __init__(
        self,
        in_channels=5,
        attr_dim=7,
        base_channels=32,
        min_sigma=0.15,
        max_sigma=0.95,
    ):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            ResidualConvBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResidualConvBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.attr_film = nn.Sequential(
            nn.Linear(attr_dim, base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 2, base_channels * 2),
        )
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, lq_img, attr_vec, latent_size):
        lq01 = (lq_img.float() + 1.0) * 0.5
        luminance = (
            0.299 * lq01[:, 0:1] + 0.587 * lq01[:, 1:2] + 0.114 * lq01[:, 2:3]
        ).clamp(0.0, 1.0)
        darkness = 1.0 - luminance
        local_mean = F.avg_pool2d(luminance, kernel_size=7, stride=1, padding=3)
        local_contrast = (luminance - local_mean).abs()
        evidence = torch.cat([lq01, darkness, local_contrast], dim=1)

        features = self.encoder(evidence)
        gamma, beta = self.attr_film(attr_vec.float()).chunk(2, dim=1)
        gamma = gamma[:, :, None, None].to(dtype=features.dtype, device=features.device)
        beta = beta[:, :, None, None].to(dtype=features.dtype, device=features.device)
        features = features * (1.0 + gamma) + beta

        sigma_logits = self.head(features)
        sigma_logits = F.interpolate(
            sigma_logits, size=latent_size, mode="bilinear", align_corners=False
        )
        sigma = torch.sigmoid(sigma_logits)
        return self.min_sigma + (self.max_sigma - self.min_sigma) * sigma




def illumination_prior_map(lq_img, latent_size, min_sigma=0.15, max_sigma=0.95):
    lq01 = (lq_img.float() + 1.0) * 0.5
    luminance = (
        0.299 * lq01[:, 0:1] + 0.587 * lq01[:, 1:2] + 0.114 * lq01[:, 2:3]
    ).clamp(0.0, 1.0)
    darkness = 1.0 - luminance
    prior = F.interpolate(darkness, size=latent_size, mode="bilinear", align_corners=False)
    return min_sigma + (max_sigma - min_sigma) * prior


class DegradationTokenEncoder(nn.Module):
    """Converts low-light degradation statistics into SD3 context tokens."""

    def __init__(self, embed_dim, num_tokens=4, base_channels=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResidualConvBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResidualConvBlock(base_channels),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(base_channels + 6, base_channels * 4),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 4, num_tokens * embed_dim),
        )
        self.token_scale = nn.Parameter(torch.tensor(0.02))

    def forward(self, lq_img):
        lq01 = (lq_img.float() + 1.0) * 0.5
        luminance = (
            0.299 * lq01[:, 0:1] + 0.587 * lq01[:, 1:2] + 0.114 * lq01[:, 2:3]
        ).clamp(0.0, 1.0)
        darkness = 1.0 - luminance
        local_mean = F.avg_pool2d(luminance, kernel_size=7, stride=1, padding=3)
        local_contrast = (luminance - local_mean).abs()
        color_mean = lq01.mean(dim=(2, 3))
        stats = torch.cat(
            [
                luminance.mean(dim=(2, 3)),
                darkness.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                color_mean,
            ],
            dim=1,
        )
        features = torch.cat([lq01, luminance, darkness, local_contrast], dim=1)
        pooled = self.stem(features).flatten(1)
        tokens = self.mlp(torch.cat([pooled, stats], dim=1))
        tokens = tokens.view(lq_img.shape[0], self.num_tokens, self.embed_dim)
        return tokens * self.token_scale


class AttributePredictionHead(nn.Module):
    """Aligns learned degradation tokens with interpretable language attributes."""

    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, len(ATTRIBUTE_SCHEMA[name])) for name in ATTRIBUTE_NAMES]
        )

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        shared = self.shared(pooled)
        return [head(shared) for head in self.heads]


def degradation_attribute_loss(attr_logits, attr_labels):
    losses = []
    for idx, logits in enumerate(attr_logits):
        losses.append(F.cross_entropy(logits.float(), attr_labels[:, idx], reduction="mean"))
    return torch.stack(losses).mean()




class FrequencySeparationLoss(nn.Module):
    """Low-frequency fidelity plus high-frequency detail supervision."""

    def __init__(self, kernel_size=21, sigma=3.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("kernel", kernel_2d[None, None])
        self.kernel_size = kernel_size

    def blur(self, x):
        kernel = self.kernel.to(dtype=x.dtype, device=x.device).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=x.shape[1])

    def forward(self, pred_img, hq_img):
        pred_low = self.blur(pred_img)
        hq_low = self.blur(hq_img)
        pred_high = pred_img - pred_low
        hq_high = hq_img - hq_low
        loss_low = F.l1_loss(pred_low, hq_low, reduction="mean")
        loss_high = F.l1_loss(pred_high, hq_high, reduction="mean")
        return loss_low, loss_high

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def set_vae_encoder_lora(vae_encoder, rank):
    target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    vae_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vae_encoder = PeftModel(
        model=vae_encoder, 
        peft_config=vae_encoder_lora_config, 
        adapter_name="vae_encoder_adapter",
        )
    vae_encoder.print_trainable_parameters()
    return vae_encoder


def set_sd3_transformer_lora(sd3_transformer, rank):
    target_modules = [
        "x_embedder",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    sd3_transformer = PeftModel(
        model=sd3_transformer, 
        peft_config=transformer_lora_config, 
        adapter_name="sd3_adapter",
        )
    sd3_transformer.print_trainable_parameters()
    return sd3_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeLang-SR.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/delang_sr_x2.yaml",
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(args.output_dir, "cfg.yml"))

    # Cast non-trainable weights to reduced precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # prompt embeds for SD3
    if not args.fixed_prompt_path:
        text_encoding_pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.sd3_path, transformer=None, vae=None, torch_dtype=weight_dtype
        )
        text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                text_encoding_pipeline.encode_prompt(
                    prompt=args.fixed_prompt,
                    prompt_2=args.fixed_prompt,
                    prompt_3=args.fixed_prompt,
                    negative_prompt="",
                    negative_prompt_2="",
                    negative_prompt_3="",
                    do_classifier_free_guidance=False,
                )
            )
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        del text_encoding_pipeline
        free_memory()
    else:
        prompts = torch.load(args.fixed_prompt_path, weights_only=True, map_location=accelerator.device)
        prompt_embeds = prompts["prompt_embeds"]
        pooled_prompt_embeds = prompts["pooled_prompt_embeds"]

    language_prompt_cache = None
    if getattr(args, "prompt_embed_cache_path", None):
        language_prompt_cache = torch.load(
            args.prompt_embed_cache_path,
            map_location="cpu",
        )
        cache_prompt_mode = language_prompt_cache.get("prompt_mode")
        if cache_prompt_mode is not None and cache_prompt_mode != getattr(args, "prompt_mode", "prompt_instruction"):
            raise ValueError(
                f"Prompt cache mode '{cache_prompt_mode}' does not match config "
                f"prompt_mode '{getattr(args, 'prompt_mode', 'prompt_instruction')}'. "
                "Regenerate the prompt cache for this config."
            )
        logger.info(
            f"Loaded DeLang prompt cache with "
            f"{language_prompt_cache['prompt_embeds'].shape[0]} unique prompts "
            f"from {args.prompt_embed_cache_path}"
        )

    # mid-timestep: SD3 uses FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.sd3_path, subfolder="scheduler")
    scheduler.set_timesteps(1000, device=accelerator.device)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    mid_idx = int(args.mid_timestep)
    mid_idx = max(0, min(mid_idx, len(timesteps) - 1))

    t_mid_tensor = timesteps[mid_idx]
    sigma_mid_tensor = sigmas[mid_idx]

    t_mid = t_mid_tensor.item() if isinstance(t_mid_tensor, torch.Tensor) else float(t_mid_tensor)
    sigma_mid = sigma_mid_tensor.item() if isinstance(sigma_mid_tensor, torch.Tensor) else float(sigma_mid_tensor)

    logger.info(
        f"Current {args.model} mid_idx={mid_idx} / {len(timesteps)-1}, "
        f"t_mid={t_mid:.6f}, sigma_mid={sigma_mid:.6f}"
    )

    prompt_embed_dim = prompt_embeds.shape[-1]

    # fixed vae
    fixed_vae = AutoencoderKL.from_pretrained(args.sd3_path, subfolder="vae")
    fixed_vae.requires_grad_(False)
    fixed_vae.eval()

    # lora_vae (frozen VAE backbone; train VAE encoder LoRA)
    lora_vae = copy.deepcopy(fixed_vae)
    lora_vae.requires_grad_(False)
    del lora_vae.decoder
    free_memory()
    lora_vae.encoder = set_vae_encoder_lora(lora_vae.encoder, args.vae_lora_rank)
    lora_vae.train()

    # sd3_transformer
    sd3_transformer = SD3Transformer2DModel.from_pretrained(
        args.sd3_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    sd3_transformer.requires_grad_(False)
    sd3_transformer = set_sd3_transformer_lora(sd3_transformer, args.transformer_lora_rank)
    sd3_transformer.train()

    # xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            sd3_transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )

    attr_dim = int(getattr(args, "attribute_vector_dim", 7))
    sigma_predictor = LanguageGuidedSigmaPredictor(
        attr_dim=attr_dim,
        base_channels=args.sigma_base_channels,
        min_sigma=args.sigma_min,
        max_sigma=args.sigma_max,
    )
    degradation_token_encoder = DegradationTokenEncoder(
        embed_dim=prompt_embed_dim,
        num_tokens=args.num_degradation_tokens,
        base_channels=args.token_base_channels,
    )
    attr_prediction_head = AttributePredictionHead(embed_dim=prompt_embed_dim)
    frequency_loss = FrequencySeparationLoss(
        kernel_size=args.freq_kernel_size,
        sigma=args.freq_kernel_sigma,
    )

    # DINOv3-ConvNeXt DISTS Loss
    from dinov3_convnext_dists import DINOv3ConvNeXtDISTS

    net_dv3d = DINOv3ConvNeXtDISTS(
        dinov3_convnext_size=args.dinov3_convnext_size,
        repo_path=args.dinov3_repo_path,
        weights_path=args.dinov3_weights_path,
    )

    fixed_vae.to(device=accelerator.device)
    lora_vae.to(device=accelerator.device)
    sd3_transformer.to(device=accelerator.device)
    sigma_predictor.to(device=accelerator.device)
    degradation_token_encoder.to(device=accelerator.device)
    attr_prediction_head.to(device=accelerator.device)
    frequency_loss.to(device=accelerator.device)
    net_dv3d.to(device=accelerator.device)

    if args.gradient_checkpointing:
        lora_vae.enable_gradient_checkpointing()
        sd3_transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(
        f"Total vae_encoder training parameters: {sum([p.numel() for p in lora_vae.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total sd3_transformer training parameters: {sum([p.numel() for p in sd3_transformer.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total sigma predictor training parameters: {sum([p.numel() for p in sigma_predictor.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total degradation token encoder training parameters: {sum([p.numel() for p in degradation_token_encoder.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total attribute alignment head training parameters: {sum([p.numel() for p in attr_prediction_head.parameters() if p.requires_grad]) / 1000000} M"
    )
    sr_opt = list(filter(lambda p: p.requires_grad, lora_vae.parameters())) + list(
        filter(lambda p: p.requires_grad, sd3_transformer.parameters())
    ) + list(
        filter(lambda p: p.requires_grad, sigma_predictor.parameters())
    ) + list(
        filter(lambda p: p.requires_grad, degradation_token_encoder.parameters())
    ) + list(
        filter(lambda p: p.requires_grad, attr_prediction_head.parameters())
    )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_sr = optimizer_class(
        sr_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = RELLISURLanguageDataset(
        args.metadata_path,
        args.resolution,
        prompt_mode=getattr(args, "prompt_mode", "prompt_instruction"),
    )
    if len(train_dataset.records[0]["attribute_vector"]) != attr_dim:
        raise ValueError(
            f"attribute_vector_dim is {attr_dim}, but metadata vectors have "
            f"{len(train_dataset.records[0]['attribute_vector'])} values."
        )
    if language_prompt_cache is not None:
        cached_records = language_prompt_cache["record_prompt_ids"].shape[0]
        if cached_records != len(train_dataset):
            raise ValueError(
                f"Prompt cache has {cached_records} records but dataset has {len(train_dataset)}. "
                "Regenerate the cache for this metadata file."
            )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        persistent_workers=False,   
        prefetch_factor=2,         
        pin_memory=False, 
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_sr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sr,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    (
        lora_vae,
        sd3_transformer,
        sigma_predictor,
        degradation_token_encoder,
        attr_prediction_head,
        optimizer_sr,
        train_dataloader,
        lr_scheduler_sr,
    ) = accelerator.prepare(
        lora_vae,
        sd3_transformer,
        sigma_predictor,
        degradation_token_encoder,
        attr_prediction_head,
        optimizer_sr,
        train_dataloader,
        lr_scheduler_sr,
    )

    # Recompute train steps after DDP wrapping
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info(f"***** Start training {args.model} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        raise ValueError(
            "Released training checkpoints contain model adapters, not optimizer "
            "or RNG state. Start a new training run with resume_from_checkpoint: null."
        )
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    vae_scale_factor = 2 ** (len(fixed_vae.config.block_out_channels) - 1)

    def expand_to_batch(tensor, batch_size):
        if tensor.shape[0] == batch_size:
            return tensor
        return tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))

    if language_prompt_cache is None:
        raise ValueError(
            "DeLang language prompts are enabled but prompt_embed_cache_path is null. "
            "Run scripts/cache_delang_prompt_embeds.py before training."
        )

    def select_prompt_context(batch_indices, device, dtype, batch_size):
        dropout = float(getattr(args, "prompt_dropout", 0.0))
        dropped = torch.rand((), device=device).item() < dropout
        if not dropped:
            prompt_ids = language_prompt_cache["record_prompt_ids"][batch_indices.cpu()].long()
            lang_prompt = language_prompt_cache["prompt_embeds"].index_select(0, prompt_ids)
            lang_pooled = language_prompt_cache["pooled_prompt_embeds"].index_select(0, prompt_ids)
            return lang_prompt.to(device=device, dtype=dtype), lang_pooled.to(device=device, dtype=dtype)

        base_prompt = expand_to_batch(prompt_embeds, batch_size).to(device=device, dtype=dtype)
        base_pooled = expand_to_batch(pooled_prompt_embeds, batch_size).to(device=device, dtype=dtype)
        return base_prompt, base_pooled

    def build_condition_context(lq_img, batch_indices):
        bsz = lq_img.shape[0]
        base_prompt, conditioned_pooled = select_prompt_context(
            batch_indices,
            device=lq_img.device,
            dtype=weight_dtype,
            batch_size=bsz,
        )
        deg_tokens = degradation_token_encoder(lq_img).to(dtype=weight_dtype)
        attr_logits = attr_prediction_head(deg_tokens)
        conditioned_prompt = torch.cat([base_prompt, deg_tokens], dim=1)
        return conditioned_prompt, conditioned_pooled, attr_logits

    def one_mid_timestep_pred(lq_latent, sigma_map, conditioned_prompt, conditioned_pooled):
        bsz, _, _, _ = lq_latent.shape

        timestep = torch.full((bsz,), t_mid, device=lq_latent.device, dtype=weight_dtype)

        # SD3 Transformer forward pass
        model_pred = sd3_transformer(
            hidden_states=lq_latent,
            timestep=timestep,
            encoder_hidden_states=conditioned_prompt,
            pooled_projections=conditioned_pooled,
            return_dict=False,
        )[0]

        # Spatially adaptive one-step ODE integration from sigma(x, y) -> 0.
        lq_latent = lq_latent - sigma_map.to(dtype=lq_latent.dtype) * model_pred

        lq_latent = (lq_latent / fixed_vae.config.scaling_factor) + fixed_vae.config.shift_factor
        pred_img = fixed_vae.decode(lq_latent.to(fixed_vae.dtype), return_dict=False)[0]
        return pred_img

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(
                lora_vae,
                sd3_transformer,
                sigma_predictor,
                degradation_token_encoder,
                attr_prediction_head,
            ):
                # Prepare data
                lq_img = batch["lq"].to(accelerator.device)
                hq_img = batch["hq"].to(accelerator.device)
                attr_vec = batch["attr_vec"].to(accelerator.device)
                attr_labels = batch["attr_labels"].to(accelerator.device)
                batch_indices = batch["index"]

                hq_latent = encode_images(hq_img, fixed_vae, weight_dtype)
                noise = torch.randn_like(hq_latent)
                sigma_map = sigma_predictor(lq_img, attr_vec, hq_latent.shape[-2:]).to(dtype=weight_dtype)
                sigma_prior = illumination_prior_map(
                    lq_img,
                    hq_latent.shape[-2:],
                    min_sigma=args.sigma_min,
                    max_sigma=args.sigma_max,
                ).to(dtype=weight_dtype)
                # Flow Matching: x_sigma = (1 - sigma) * x0 + sigma * epsilon, where sigma is spatially adaptive.
                pretrained_noisy_latent = (1 - sigma_map) * hq_latent + sigma_map * noise

                lq_latent = encode_images(lq_img, unwrap_model(lora_vae), weight_dtype)
                conditioned_prompt, conditioned_pooled, attr_logits = build_condition_context(
                    lq_img,
                    batch_indices,
                )

                # LRR Loss: Latent Representation Refinement Loss
                loss_LRR = F.mse_loss(pretrained_noisy_latent, lq_latent, reduction="mean") * args.lambda_LRR

                # Onestep prediction at mid-timestep
                pred_img = one_mid_timestep_pred(
                    lq_latent,
                    sigma_map,
                    conditioned_prompt,
                    conditioned_pooled,
                )

                # DINOv3-ConvNext DISTS Loss
                loss_Dv3D = net_dv3d(pred_img, hq_img) * args.lambda_Dv3D

                # L1 Loss
                loss_L1 = F.l1_loss(pred_img, hq_img, reduction="mean") * args.lambda_L1

                # Illumination prior keeps darker regions assigned to stronger flow correction.
                loss_sigma_prior = F.l1_loss(
                    sigma_map.float(), sigma_prior.float(), reduction="mean"
                ) * args.lambda_sigma_prior

                # Frequency separation: low-frequency color fidelity and high-frequency detail recovery.
                loss_freq_low_raw, loss_freq_high_raw = frequency_loss(pred_img, hq_img)
                loss_freq_low = loss_freq_low_raw * args.lambda_freq_low
                loss_freq_high = loss_freq_high_raw * args.lambda_freq_high

                loss_attr = degradation_attribute_loss(attr_logits, attr_labels) * args.lambda_attr
                
                total_G_loss = (
                    loss_LRR
                    + loss_Dv3D
                    + loss_L1
                    + loss_sigma_prior
                    + loss_freq_low
                    + loss_freq_high
                    + loss_attr
                )

                accelerator.backward(total_G_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sr_opt, args.max_grad_norm)

                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad()

            if accelerator.sync_gradients:
                if accelerator.is_main_process and global_step % args.save_img_steps == 0:
                    img_path = os.path.join(args.output_dir, f"img-{global_step}.png")
                    sigma_vis = F.interpolate(
                        sigma_map[0:1].float(),
                        size=pred_img.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).repeat(1, 3, 1, 1)
                    save_imgs = torch.cat(
                        [
                            (torch.stack([lq_img[0], pred_img[0], hq_img[0]], dim=0) + 1) / 2,
                            sigma_vis.detach(),
                        ],
                        dim=0,
                    )
                    save_image(save_imgs.detach(), img_path)
                    # logger.info(f"img-{global_step}.png saved!")

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    if global_step % args.checkpointing_steps == 0:
                        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
                        os.makedirs(weight_path, exist_ok=True)
                        unwrap_model(sd3_transformer).save_pretrained(weight_path)
                        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
                        torch.save(
                            {
                                "sigma_predictor": unwrap_model(sigma_predictor).state_dict(),
                                "degradation_token_encoder": unwrap_model(degradation_token_encoder).state_dict(),
                                "attr_prediction_head": unwrap_model(attr_prediction_head).state_dict(),
                                "attribute_schema": ATTRIBUTE_SCHEMA,
                                "module_config": {
                                    "attribute_vector_dim": attr_dim,
                                    "sigma_mode": "language",
                                    "use_language_sigma": True,
                                    "use_language_prompt": True,
                                    "use_visual_degradation_tokens": True,
                                    "use_attr_alignment": True,
                                    "num_degradation_tokens": int(args.num_degradation_tokens),
                                    "sigma_min": float(args.sigma_min),
                                    "sigma_max": float(args.sigma_max),
                                },
                            },
                            os.path.join(weight_path, "delang_modules.pt"),
                        )
                        logger.info(f"Saved weight to {weight_path}")

            logs = {
                "loss_LRR": loss_LRR.detach().item(),
                "loss_Dv3D": loss_Dv3D.detach().item(),
                "loss_L1": loss_L1.detach().item(),
                "loss_sigma_prior": loss_sigma_prior.detach().item(),
                "loss_freq_low": loss_freq_low.detach().item(),
                "loss_freq_high": loss_freq_high.detach().item(),
                "loss_attr": loss_attr.detach().item(),
                "sigma_mean": sigma_map.detach().float().mean().item(),
                "lr": lr_scheduler_sr.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
        os.makedirs(weight_path, exist_ok=True)
        unwrap_model(sd3_transformer).save_pretrained(weight_path)
        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
        torch.save(
            {
                "sigma_predictor": unwrap_model(sigma_predictor).state_dict(),
                "degradation_token_encoder": unwrap_model(degradation_token_encoder).state_dict(),
                "attr_prediction_head": unwrap_model(attr_prediction_head).state_dict(),
                "attribute_schema": ATTRIBUTE_SCHEMA,
                "module_config": {
                    "attribute_vector_dim": attr_dim,
                    "sigma_mode": "language",
                    "use_language_sigma": True,
                    "use_language_prompt": True,
                    "use_visual_degradation_tokens": True,
                    "use_attr_alignment": True,
                    "num_degradation_tokens": int(args.num_degradation_tokens),
                    "sigma_min": float(args.sigma_min),
                    "sigma_max": float(args.sigma_max),
                },
            },
            os.path.join(weight_path, "delang_modules.pt"),
        )
        logger.info(f"Saved weight to {weight_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
