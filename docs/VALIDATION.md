# Release preparation validation

Date: 2026-09-26.

## Completed

- The released model/trainer file is byte-identical to the inspected active
  training checkout. SHA256:
  `ecb4ff6bc99d724248e8215a0c435e8e30cca4e96a122588317ed11a13bb1245`.
- All Python files parse, `runtime.json` parses, and README Bash examples pass
  `bash -n`.
- Installer test passes: first install, identical repeat, and refusal to
  overwrite a different trainer while preserving its contents.
- In the original training container, installed-source checking, trainer
  discovery, and construction for first strides `(2, 2, 2)` and `(1, 2, 2)`
  pass. The container emitted dependency deprecation warnings and a teardown
  warning/bus-error message after the Python checks; no GPU was available.
- Official nnunetv2 2.6.2 was downloaded from PyPI into an isolated temporary
  package directory. Installation of this release's trainer, installed-source
  comparison, nnU-Net class discovery, and model construction passed there.
  The check also passed with official dynamic-network-architectures 0.4.1
  installed in that temporary directory. Other container dependencies were reused; this was not a clean full
  dependency installation. See below for remaining checks.
- For synthetic model settings (one input channel, four output channels,
  level0 kernel 3), parameter counts were 6,028,020 for stride `(2, 2, 2)` and
  6,021,748 for `(1, 2, 2)`. These are not the dataset-specific counts recorded
  in `runtime.json`.

## Dependency correction

The local training checkout identifies as nnunetv2 2.6.1, but that version is
not published on PyPI. New installations therefore target official 2.6.2.
Its metadata requires dynamic-network-architectures >=0.4.1,<0.5, so the new
installation file pins 0.4.1. Historical runtime records retain 0.3.1 and the
local nnU-Net version rather than pretending the experiment used this new stack.

## Remaining checks

- Complete fresh installation of all dependencies on the intended CUDA system
  and `python -m pip check`.
- `python smoke_test.py --cuda` for forward/backward execution and finite
  gradients. The current preparation node has no available CUDA GPU.
- A dataset-specific training/prediction integration run on official nnU-Net,
  including multiple GPUs when used. Construction checks do not establish
  training equivalence between the local checkout and official nnU-Net.
- Existing `runtime.json` reference-comparison results are historical; they
  were not regenerated in this preparation.

No running training job, training source file, dataset, or checkpoint was
modified by these checks. No weights are included.

## ACDC checkpoint release

The packaged ACDC best checkpoint preserves the selected network weights,
strictly loads into the public SlimFormer++ model, and initializes through the
official nnunetv2 2.6.2 predictor on CPU. GPU prediction of the packaged
asset has not been run in this preparation environment.

## Attention variant

Date: 2026-09-28. `nnUNetTrainer_SlimFormerPlusPlus_Attention.py` is a copy of
the released model file with the two encoder-1 `SelectiveMixer` modules replaced
by `WindowSelfAttention`. The released SlimFormer++ file is unchanged.

- Structure: for the ACDC, AbdomenCT-1K and AMOS2022 plans, all 178 tensors
  outside the encoder-1 token mixers have the same names and shapes as in
  SlimFormer++. With the released best checkpoints loaded into those tensors
  and the encoder-1 token-mixer residual scale set to zero in both models, the
  two models give bitwise-identical outputs on a random patch (CUDA, FP32).
- A fresh Python 3.12 virtual environment on Linux aarch64 (NVIDIA GH200)
  ran the README attention installation block verbatim: torch 2.6.0+cu126,
  torchvision 0.21.0, nnunetv2 2.6.2, `pip check` clean, `install.py
  --attention`, and `smoke_test.py --attention --cuda` passed.
  `mamba_ssm`, `causal_conv1d` and `transformers` were not installed.
- Official nnunetv2 2.6.2 trained the attention trainer on ACDC fold 0 for
  24 epochs on one GPU (plans batch 4) without errors; the run was then
  stopped. This checks that training runs, not final accuracy.
- No checkpoint of this variant has been trained to completion or released.
