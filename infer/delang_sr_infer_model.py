import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler


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




class DegradationTokenEncoder(nn.Module):
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




def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype: torch.dtype) -> torch.Tensor:
    """
    Encode RGB tensors into SD3 latents with shift+scaling applied.
    """
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


class DeLang_SR_Infer(torch.nn.Module):
    """
    One-step SR inference for SD3 + FlowMatch, with optional latent tiling.
    """

    def __init__(
        self,
        sd3_path: str,
        lora_path: str,
        device: str,
        prompt_embed_dim: int,
        weight_dtype: torch.dtype = torch.bfloat16,
        mid_timestep: int = 700,
        sigma_min: float = 0.15,
        sigma_max: float = 0.95,
        sigma_base_channels: int = 32,
        num_degradation_tokens: int = 4,
        token_base_channels: int = 64,
        attribute_vector_dim: int = 7,
    ):
        super().__init__()

        vae = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae")
        transformer = SD3Transformer2DModel.from_pretrained(sd3_path, subfolder="transformer")
        sigma_predictor = LanguageGuidedSigmaPredictor(
            attr_dim=attribute_vector_dim,
            base_channels=sigma_base_channels,
            min_sigma=sigma_min,
            max_sigma=sigma_max,
        )
        degradation_token_encoder = DegradationTokenEncoder(
            embed_dim=prompt_embed_dim,
            num_tokens=num_degradation_tokens,
            base_channels=token_base_channels,
        )

        # FlowMatch scheduler to fetch sigma/timestep at the chosen mid-timestep
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sd3_path, subfolder="scheduler")
        scheduler.set_timesteps(1000, device=device)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas
        mid_idx = max(0, min(int(mid_timestep), len(timesteps) - 1))
        self.t_mid = timesteps[mid_idx]
        self.sigma_mid = sigmas[mid_idx]
        print(f"[DeLang-SR] mid_timestep idx={mid_idx}, t_mid={float(self.t_mid):.6f}, scheduler_sigma={float(self.sigma_mid):.6f}")

        # Load LoRA adapter for transformer
        transformer = PeftModel.from_pretrained(transformer, os.path.join(lora_path, "sd3_adapter"), is_trainable=False)
        transformer = transformer.merge_and_unload()

        # Optional: Load VAE encoder adapter if present (for SD3 runs that train VAE encoder LoRA)
        vae_encoder_adapter_path = os.path.join(lora_path, "vae_encoder_adapter")
        if os.path.exists(vae_encoder_adapter_path):
            print("Loading VAE encoder adapter...")
            vae.encoder = PeftModel.from_pretrained(vae.encoder, vae_encoder_adapter_path, is_trainable=False)
            vae.encoder = vae.encoder.merge_and_unload()
        else:
            raise FileNotFoundError(f"Missing VAE encoder adapter: {vae_encoder_adapter_path}")

        module_path = os.path.join(lora_path, "delang_modules.pt")
        if os.path.exists(module_path):
            print("Loading DeLang sigma predictor and degradation token encoder...")
            module_state = torch.load(module_path, map_location="cpu", weights_only=True)
            module_config = module_state.get("module_config", {})
            if module_config.get("use_language_sigma", True) is False:
                raise ValueError("The checkpoint is not a full DeLang-SR model.")
            if module_config.get("attribute_vector_dim", attribute_vector_dim) != attribute_vector_dim:
                raise ValueError("Checkpoint descriptor dimension does not match inference.")
            sigma_predictor.load_state_dict(module_state["sigma_predictor"])
            degradation_token_encoder.load_state_dict(module_state["degradation_token_encoder"])
        else:
            raise FileNotFoundError(f"DeLang module checkpoint not found: {module_path}")

        vae.requires_grad_(False)
        transformer.requires_grad_(False)
        sigma_predictor.requires_grad_(False)
        degradation_token_encoder.requires_grad_(False)
        vae.to(device=device, dtype=weight_dtype)
        transformer.to(device=device, dtype=weight_dtype)
        sigma_predictor.to(device=device)
        degradation_token_encoder.to(device=device)
        vae.eval()
        transformer.eval()
        sigma_predictor.eval()
        degradation_token_encoder.eval()

        self.vae = vae
        self.transformer = transformer
        self.sigma_predictor = sigma_predictor
        self.degradation_token_encoder = degradation_token_encoder
        self.device = device
        self.weight_dtype = weight_dtype
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _sync_device(self):
        if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _expand_to_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        return tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))

    def _build_condition_context(self, lq_img: torch.Tensor, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor):
        bsz = lq_img.shape[0]
        prompt_embeds = self._expand_to_batch(prompt_embeds, bsz).to(device=lq_img.device, dtype=self.weight_dtype)
        pooled_prompt_embeds = self._expand_to_batch(pooled_prompt_embeds, bsz).to(device=lq_img.device, dtype=self.weight_dtype)
        deg_tokens = self.degradation_token_encoder(lq_img.float()).to(dtype=self.weight_dtype)
        prompt_embeds = torch.cat([prompt_embeds, deg_tokens], dim=1)
        return prompt_embeds, pooled_prompt_embeds

    def _gaussian_weights(self, tile_width: int, tile_height: int, nbatches: int) -> torch.Tensor:
        """
        Generate a Gaussian mask for blending tiled latent predictions.
        """
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.transformer.config.in_channels, 1, 1))

    def _run_transformer(self, latents: torch.Tensor, sigma_map: torch.Tensor, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> torch.Tensor:
        bsz = latents.shape[0]
        timestep = torch.full((bsz,), float(self.t_mid), device=latents.device, dtype=self.weight_dtype)
        model_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        latents = latents - sigma_map.to(dtype=latents.dtype) * model_pred
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        pred_img = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        return pred_img.clamp(-1, 1)

    def _forward_no_tile(self, lq_latent: torch.Tensor, sigma_map: torch.Tensor, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> torch.Tensor:
        return self._run_transformer(lq_latent, sigma_map, prompt_embeds, pooled_prompt_embeds)

    def _forward_tile(self, lq_latent: torch.Tensor, sigma_map: torch.Tensor, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor, tile_size: int, tile_overlap: int) -> torch.Tensor:
        _, _, h, w = lq_latent.shape
        tile_size = min(tile_size, min(h, w))
        tile_overlap = min(tile_overlap, tile_size - 1)
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        # Compute grid
        grid_rows = 0
        cur_x = 0
        while cur_x < w:
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < h:
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list = []
        sigma_list = []
        noise_preds = []
        sigma_tiles = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_tile = lq_latent[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
                sigma_tile = sigma_map[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
                input_list.append(input_tile)
                sigma_list.append(sigma_tile)

                if len(input_list) == 1 or col == grid_cols - 1:
                    input_cat = torch.cat(input_list, dim=0)
                    prompt_cat = self._expand_to_batch(prompt_embeds, input_cat.shape[0]).to(self.weight_dtype)
                    pooled_cat = self._expand_to_batch(pooled_prompt_embeds, input_cat.shape[0]).to(self.weight_dtype)
                    model_out = self.transformer(
                        hidden_states=input_cat,
                        timestep=torch.full((input_cat.shape[0],), float(self.t_mid), device=input_cat.device, dtype=self.weight_dtype),
                        encoder_hidden_states=prompt_cat,
                        pooled_projections=pooled_cat,
                        return_dict=False,
                    )[0]
                    sigma_tiles.extend(sigma_list)
                    input_list = []
                    sigma_list = []
                noise_preds.append(model_out)

        # Stitch predictions
        noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=self.weight_dtype)
        contributors = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=self.weight_dtype)
        idx = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size
                noise_pred[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += (
                    sigma_tiles[idx].to(dtype=self.weight_dtype) * noise_preds[idx] * tile_weights
                )
                contributors[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += tile_weights
                idx += 1

        noise_pred = noise_pred / contributors
        latents = lq_latent - noise_pred
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        pred_img = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        return pred_img.clamp(-1, 1)

    def forward(
        self,
        lq_img: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        attr_vec: torch.Tensor,
        tile_size: int,
        tile_overlap: int,
        return_intermediates: bool = False,
    ):
        self._sync_device()
        start_time = time.time()

        lq_latent = encode_images(lq_img.to(self.vae.dtype), self.vae, self.weight_dtype)
        sigma_map = self.sigma_predictor(lq_img.float(), attr_vec.to(lq_img.device), lq_latent.shape[-2:])
        prompt_embeds, pooled_prompt_embeds = self._build_condition_context(lq_img, prompt_embeds, pooled_prompt_embeds)
        _, _, h, w = lq_latent.shape
        if h <= tile_size and w <= tile_size:
            print("[DeLang-SR] small input, no tiling.")
            pred_img = self._forward_no_tile(lq_latent, sigma_map, prompt_embeds, pooled_prompt_embeds)
        else:
            print(f"[DeLang-SR] tiled latent inference, size {lq_img.shape[-2]}x{lq_img.shape[-1]}")
            pred_img = self._forward_tile(lq_latent, sigma_map, prompt_embeds, pooled_prompt_embeds, tile_size, tile_overlap)

        self._sync_device()
        t = time.time() - start_time
        print(f"[DeLang-SR] inference time per image: {t}s, sigma_mean={sigma_map.float().mean().item():.4f}")
        if return_intermediates:
            return pred_img, t, {"sigma_map": sigma_map.detach().float()}
        return pred_img, t
