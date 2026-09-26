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

The ACDC `ep1200-b8` run completed 1200 epochs. Its selected best network
weights (epoch 254) strictly load into the public SlimFormer++ model. A packaged
inference-only copy preserves every weight tensor and loads through the official
nnunetv2 2.6.2 predictor initialization on CPU. Full GPU inference from the
packaged asset has not been run in this preparation environment. The reported
0.916 foreground mean Dice is from 60 cases in the original fold-0 validation
set, not an independent test set.
