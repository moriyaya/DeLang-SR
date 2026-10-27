# Third-party code and models

DeLang-SR builds on the SD3 inference/training infrastructure and DINOv3-ConvNeXt
feature-distance implementation used in [OMGSR](https://github.com/wuer5/OMGSR).
The adaptation adds input-derived degradation conditions, visual degradation
tokens, descriptor-guided spatial update strength, frequency supervision, and
attribute alignment. Upstream infrastructure and losses are credited here;
removing experimental scripts does not remove this provenance.

- **Hugging Face Diffusers**: the training script retains the original
  Hugging Face copyright and Apache-2.0 notice. See
  <https://github.com/huggingface/diffusers>.
- **OMGSR**: parts of the training/inference scaffolding, latent tile blending,
  and `dinov3_convnext_dists.py` are adapted from the upstream implementation.
  No root license file was present in the upstream snapshot inspected for this
  release. This repository does not grant additional rights to those portions;
  consult the upstream authors for reuse or redistribution terms.
- **Stable Diffusion 3 Medium**: download the base model separately from
  <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>.
  Follow the model's access conditions and license. Base model and text encoder
  weights are not redistributed here.
- **DINOv3**: training uses the official local repository and pretrained
  ConvNeXt-Large weights. Download them separately and follow the terms at
  <https://github.com/facebookresearch/dinov3>. They are not required for inference.
- **PyIQA**: optional evaluation uses the official implementations from
  <https://github.com/chaofengc/IQA-PyTorch>, under their respective terms.
- **RELLISUR**: images on the project page are benchmark examples with attribution
  to Aakerberg et al., *RELLISUR: A Real Low-Light Image Super-Resolution Dataset*,
  NeurIPS Datasets and Benchmarks 2021. Obtain benchmark data from its authors.

## Licensing scope

The retained third-party files and model artifacts remain subject to their
respective notices and terms. No blanket license or permission for commercial
use of the complete pipeline is asserted by this release. DeLang-SR adapters
depend on Stable Diffusion 3 Medium and do not replace its model license.

