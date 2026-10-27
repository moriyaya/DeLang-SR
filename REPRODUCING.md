# Reproduction and evaluation protocol

## Models and inputs

Use the full-model 20K adapters for the matching x2 or x4 task. The pretrained
base is SD3 Medium; SD3.5 or other diffusion priors are not interchangeable.
The latent update uses four visual tokens, a seven-dimensional descriptor,
index 700 of a 1000-entry FlowMatch scheduler, and strength bounds [0.15, 0.95].

By default, inference builds conditions from the entire supplied LQ input set.
Fixed exposure/contrast proxies and RGB statistics are combined with input-set
quantiles for noise, detail, and illumination labels. No HQ image is read.
Consequently, changing the input set can change quantile-based labels. For
benchmark comparison, supply the whole test split in one invocation, or build
its input-only metadata once and reuse `--metadata_path` for every invocation.
Do not generate each test image's metadata independently and assume the same
condition calibration.

The original benchmark image preparation is available as `--preprocess paper`:
upscale with Pillow bicubic, enlarge very small inputs to the configured process
size, and align dimensions to the VAE grid with Lanczos. Latents larger than the
process window are tiled with Gaussian blending. The released entry point also
guards overlap for narrow images. `--preprocess native` instead preserves exactly
x2/x4 output dimensions with padding/cropping; it is an explicit alternative
for arbitrary images, not the preprocessing used to produce the reported table.

The VAE samples latents. A fixed seed (123 by default) is recorded for repeatable
runs; numerical results can vary with hardware, precision, seed, and dependency
versions. Inference in the release is validated on an available CUDA server.
The validation server used PyTorch 2.11.0+cu126, torchvision 0.26.0+cu126,
Diffusers 0.34.0, Transformers 4.55.2, PEFT 0.17.0, and Accelerate 1.1.0.
Both released checkpoints passed end-to-end single-image CUDA inference in this
environment. The x2 test produced 1248x1248 pixels and the x4 test 2496x2496
pixels from the same 624x624 LQ image. This smoke test verifies loading and output
geometry; it does not replace rerunning the complete 425-image benchmark.

## Metrics

Use aligned RGB outputs and GT with the same dimensions. The default evaluator
uses scikit-image PSNR/SSIM on RGB floats in [0, 1], with no border shave. Y-channel
PSNR/SSIM are saved separately. LPIPS and DISTS use PyIQA implementations.
ExpErr is the absolute difference of global mean luminance (BT.601 coefficients
0.299, 0.587, 0.114); ColorErr is the L1 sum of the three mean RGB differences.
Do not confuse this ColorErr with Lab Delta-E or an average RGB L1 pixel error.

NIQE/MUSIQ/MANIQA/CLIP-IQA use the output alone. GT is optional for pairing checks,
not a metric input. The evaluators report image counts, missing GT, and mismatched
shapes. For RELLISUR, require 425 valid pairs, zero missing GT, and zero shape
mismatches. Never enable prediction resizing to silently repair a mismatched
evaluation. Keep metric version, precision, and input resizing fixed across methods.

## Runtime

The per-image `model_time_s` in `run.json` covers the restoration model forward
pass with CUDA synchronization. It excludes text pre-encoding, file I/O, loading,
and IQA; it does not include a warm-up protocol and is not by itself a rigorous
benchmark. For a runtime table, warm up the same model/input shape and then time
synchronized repeats under identical hardware, precision, and output resolution.

## Provenance

The public tree contains the full-method code, portable configs, checkpoint
download/packaging tools, evaluators, and a static project page. Experiment logs,
internal runbooks, reviewer correspondence, intermediate results, and ablation
variants are not included. Third-party attributions are retained separately.
