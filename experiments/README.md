# Reproduce the released results

These are **fold-0 validation results**, not independent test-set results.
They use the three released best checkpoints from the completed 1,200-epoch,
global-batch-8 runs. Their weights and plans were checked against the original runs.

| Dataset argument | Train / validation volumes | Dice (%) | IoU (%) | HD95 (mm) | ASD (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `acdc` | 240 / 60 | 91.60 | 85.05 | 3.30 | 0.69 |
| `abdomenct1k` | 800 / 200 | 95.49 | 91.83 | 2.89 | 0.67 |
| `amos-ct` | 200 / 100 | 90.40 | 84.14 | 10.01 | 6.38 |

Each dataset folder contains the exact plans, dataset metadata, split, short
protocol, per-case metrics, aggregate metrics and recorded parameter/FLOP counts.
Dice and IoU in CSV/JSON are fractions. Metrics exclude background, average over
cases per label, then over labels; distance metrics follow `reproduce.py`.
ACDC has 120 training and 30 validation patients, with no patient overlap.

## Evaluate a checkpoint

Follow the repository installation instructions first. Obtain the data through
the [official sources](../docs/DATASETS.md); images and labels are not redistributed.
Point the command at your images and matching reference labels:

```bash
export nnUNet_results=/path/to/checkpoints
python reproduce.py acdc --images /path/to/images --labels /path/to/labels --output /path/to/evaluation
```

Replace `acdc` with either other dataset argument. The script selects only the
published validation cases, downloads the matching checkpoint and prints measured
versus reference metrics. Output includes `metrics.csv`, `metrics.json` and
`comparison.json`. Use a new output directory for each run.

Inputs can use the published case IDs with nnU-Net's `_0000.nii.gz` image suffix.
Native ACDC `patient*_frame*.nii.gz` images and `_gt.nii.gz` labels are also accepted;
directories are searched recursively. Label values must match `dataset.json`.
For AMOS, supply CT images only. Every validation case and its label must be present.

To evaluate predictions you already have, without GPU inference:

```bash
python reproduce.py acdc --predictions /path/to/predictions --labels /path/to/labels --output /path/to/evaluation
```

## Train from scratch

Use fresh nnU-Net directories and supply all training **and** validation volumes:

```bash
export nnUNet_raw=/path/to/raw
export nnUNet_preprocessed=/path/to/preprocessed
export nnUNet_results=/path/to/training-results
python reproduce.py acdc --train --images /path/to/images --labels /path/to/labels --gpus 4
```

This links the inputs, installs the published plans/split, preprocesses and trains.
The reference uses four GPUs, global batch 8 and 1,200 epochs. A different GPU count
must divide batch 8 evenly. Full settings are in each `protocol.json`.
The historical training environment is in [runtime.json](../runtime.json); the
supported public installation is in [VALIDATION.md](../docs/VALIDATION.md).
Retraining is not guaranteed to be bitwise identical across environments.

## Measure efficiency

```bash
python benchmark.py acdc --output /path/to/benchmark.json
```

Run once per dataset in a fresh process on a CUDA GPU. This records parameters,
GFLOPs, all timing samples, median/p95 latency, peak allocated memory and runtime
versions. Measurement uses batch 1, the dataset's actual patch, BF16 autocast and
eager evaluation, with 10 warmup and 50 measured iterations.
It measures patch forward execution; complete-volume inference includes more work.
FLOPs use dispatcher counts plus analytic Mamba kernel work; multiply-accumulates
count as two operations, and elementwise operations are excluded.
Energy measurements are not included in this release.

Fresh measurements on 2026-10-21, NVIDIA GH200 120GB:

| Dataset | Params (M) | GFLOPs | Median latency (ms) | Peak allocated memory (GiB) |
| --- | ---: | ---: | ---: | ---: |
| ACDC | 6.0027 | 328.68 | 20.30 | 0.804 |
| AbdomenCT-1K | 6.0281 | 528.90 | 29.03 | 1.017 |
| AMOS CT | 6.0031 | 545.75 | 32.51 | 1.097 |

Each `benchmark.json` contains the 50 raw timing samples and exact environment.
These use dataset-specific patches and are separate from the README comparison figures.
