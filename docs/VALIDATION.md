[← Project home](../README.md) · [Documentation](README.md)

# Validation

## Original model: fresh installation and CUDA execution

Validated on 2026-10-06 in a new Python virtual environment without system
site packages. Model code and checkpoint parameter names were unchanged.

| Component | Tested value |
| --- | --- |
| Platform / GPU | Linux aarch64 / NVIDIA GH200 120GB |
| Python | 3.12.3 |
| pip / setuptools | 24.0 / 78.1.0 |
| PyTorch / torchvision | Official 2.6.0+cu126 / 0.21.0 |
| CUDA Toolkit / driver | 12.6.85 / 565.57.01 |
| Host compiler | GCC 13.2.0 |
| nnU-Net / dynamic-network-architectures | 2.6.2 / 0.4.1 |
| Mamba / causal-conv1d | 2.2.2 / 1.4.0, built from complete GitHub sources |
| Transformers / C++11 ABI | 4.46.3 / TRUE |

Passed:

- Fresh dependency installation with `--no-build-isolation` and `MAX_JOBS=4`,
  repeated with the current pinned source commits and dependency constraints.
  Installing the remaining requirements took approximately 8–12 minutes.
- `python -m pip check`: no broken requirements.
- Trainer installation and nnU-Net class discovery.
- CUDA forward/backward for first strides `(2,2,2)` and `(1,2,2)`, with
  FP16 autocast, expected output dimensions, finite loss, and finite gradients
  for every trainable parameter.
- Imports of the compiled selective-scan and causal-conv1d extensions.
- Preflight checks for the valid environment and early rejection of missing
  CUDA Toolkit, mismatched Toolkit versions, and bundled torch library conflicts.

The installation needed two NVIDIA-container environment corrections:
disabling an unavailable preset package index and removing the image's old
torch library paths from `LD_LIBRARY_PATH`. These are documented in
[INSTALLATION.md](INSTALLATION.md). PyPI source-build failures were also
reproduced for both pinned extension versions: their archives omitted `csrc`.

[tools/constraints.txt](../tools/constraints.txt) records the tested dependency snapshot.
Triton is constrained separately on aarch64 because PyTorch's dependency
metadata differs between aarch64 and x86_64.

## Model and checkpoint checks

The released original model/trainer matches the inspected training source.
SHA256:
`ecb4ff6bc99d724248e8215a0c435e8e30cca4e96a122588317ed11a13bb1245`.

For synthetic one-input-channel/four-output-channel models with level0 kernel
3, parameter counts are 6,028,020 for stride `(2,2,2)` and 6,021,748 for
`(1,2,2)`. Dataset-specific settings are recorded in
[runtime.json](../experiments/runtime.json).

The packaged ACDC best checkpoint strictly loads into the public model and
initializes through the official nnunetv2 2.6.2 predictor on CPU. Earlier
reference comparisons in `runtime.json` are historical records, not results
of the new installation check.

The original training checkout identifies as nnunetv2 2.6.1 with
dynamic-network-architectures 0.3.1 and NVIDIA's PyTorch 2.6 prerelease.
The public installation uses official nnunetv2 2.6.2 and
dynamic-network-architectures 0.4.1. `runtime.json` preserves the original
versions; it is not an installation lock file.

## Mamba-free implementation

Fresh installation and CUDA checks were repeated on 2026-10-06 with the
current constraints, official PyTorch 2.6.0+cu126, torchvision 0.21.0 and
nnunetv2 2.6.2 in a separate Python 3.12 environment on GH200:

- Installation, `pip check`, trainer discovery and CUDA forward/backward passed.
- Mamba, causal-conv1d and Transformers were absent.

Earlier checks on 2026-09-28:

- ACDC fold 0 training ran for 24 epochs on one GPU without errors.
- The 178 tensors outside the encoder-1 token mixers retain the original
  names and shapes. With mixer residual scales set to zero and identical
  loaded weights, both models produced bitwise-identical CUDA FP32 outputs
  on a random patch.

No completed-training checkpoint of this variant has been released.

## Scope

The installation checks establish execution on the tested system. They do not
establish full training equivalence with the original checkout or multi-GPU
equivalence.

Source CI checks syntax, safe installation, experiment-bundle consistency and
evaluation on synthetic images. It does not run CUDA builds or GPU training.

## Reproduction bundles and patch benchmarks (2026-10-21)

The [three experiment bundles](../experiments/README.md) publish the original
fold-0 splits, plans, protocols and per-case reference metrics for the released
best checkpoints. All three released checkpoints were compared tensor by tensor
with the completed `slimformer-plusplus-ep1200-b8` runs; their weights, plans and
dataset metadata match. Exported aggregates match the bundled per-case metrics;
the result tables report the paper's Table III Dice and HD95 values separately.

The evaluator was checked against saved predictions for one ACDC case, one
AbdomenCT-1K case and two AMOS cases, including an empty-mask case. All metrics
matched the reference within 1e-9. Synthetic tests cover geometry, anisotropic
spacing, empty masks and selection of the exact validation cases.

New GH200 120GB benchmarks measured all three models with batch 1, their actual
dataset patches, BF16 autocast and eager evaluation: 10 warmup and 50 measured
iterations per dataset, in separate processes. Each `benchmark.json` preserves
all timing samples, software versions and source hashes. Parameter and GFLOP
counts match the historical records. Energy was not measured.

These additions did not repeat full training or full-dataset GPU prediction.

## Linux x86_64 and RTX 3090 (2026-10-24)

A fresh installation following the README on Linux x86_64 with two NVIDIA RTX
3090 24GB GPUs (Python 3.12.15, CUDA Toolkit 12.6.85, GCC 9.4.0, driver
570.211.01) passed `pip check` and the CUDA smoke tests for both models. With
the released ACDC checkpoint, `reproduce.py acdc` matched all four reference
metrics within 4e-5. `benchmark.py` parameter and GFLOP counts matched for all
three datasets. From fresh directories, `reproduce.py acdc --train --gpus 2`
extracted the fingerprint, preprocessed and trained five epochs (about 157 s
each) before it was stopped.
