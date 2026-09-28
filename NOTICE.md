# Code provenance and third-party notices

The standalone model consolidates project SlimFormer components: window
partition, MLP/LayerScale, dense window Mamba, aligned MedNeXt decoding,
plans-aware downsampling, refined-trunk routing and subpixel upsampling.
AugmentedMixer is the project-owner-supplied R5 CapFree implementation.
SelectiveMixer is the descriptive name for DenseWindowMamba with the same
computation. Encoder2/3 retain MLP residuals but no token-mixer residuals.
The release model file matches the currently inspected training source.
The attention variant (`nnUNetTrainer_SlimFormerPlusPlus_Attention.py`) is a
copy of that file in which SelectiveMixer is replaced by window self-attention
using PyTorch's `scaled_dot_product_attention`; it uses neither Mamba nor
causal-conv1d.

The source checkout's Apache-2.0 license text, including its attribution to
Division of Medical Image Computing, German Cancer Research Center (DKFZ),
Heidelberg, Germany (2019), is preserved verbatim in
`licenses/Apache-2.0.txt`. The project research-only license does not replace
third-party licenses or withdraw permissions granted by upstream owners.
Original project contributions: Copyright (c) 2026 WHHHHHY822.

External implementations remain dependencies; their source is not vendored:

- [PyTorch](https://github.com/pytorch/pytorch)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [einops](https://github.com/arogozhnikov/einops)
- [Mamba](https://github.com/state-spaces/mamba)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), source revision
  `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)

Each dependency retains its own license and notices. Consult the exact
installed revision for its terms. No dataset or pretrained weights are
included in this source release.
