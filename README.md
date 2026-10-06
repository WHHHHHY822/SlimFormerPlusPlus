# SlimFormer++

SlimFormer++ is a 3D medical image segmentation network for nnU-Net v2.
The model and trainer are in [one file](nnUNetTrainer_SlimFormerPlusPlus.py).

## Performance and efficiency

<p align="center">
  <img src="assets/performance_efficiency_pruning.png" alt="Segmentation performance versus computational cost" width="53%" />
  <img src="assets/efficiency_radar.png" alt="Normalized efficiency comparison" width="44%" />
</p>

Left: segmentation performance versus computational cost. Right: normalized
comparison of model size, memory use and runtime.

## Installation

Use **Linux, an NVIDIA GPU, Python 3.12, CUDA Toolkit 12.6 (`nvcc`),
Git and a C++ compiler**. The reference profile uses PyTorch 2.6.0 and
torchvision 0.21.0. Mamba and causal-conv1d compile from complete, pinned
GitHub sources; allow 10–20 minutes for compilation.

Use a fresh environment. For NVIDIA containers, other platforms or installation
errors, see the [installation guide](docs/INSTALLATION.md).
RTX 3090/4090 compatibility is expected but untested; RTX 5090/Blackwell
requires a separate environment and is outside this reference profile.

```bash
git clone https://github.com/WHHHHHY822/SlimFormerPlusPlus.git
cd SlimFormerPlusPlus
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install -c constraints.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c constraints.txt pip setuptools wheel packaging ninja
python check_environment.py --cuda
MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python install.py
python smoke_test.py --cuda
```

Validated on Linux aarch64 / NVIDIA GH200: fresh dependency installation,
`pip check`, trainer discovery, and CUDA forward/backward all passed.
[Validation details and limits](docs/VALIDATION.md).

`install.py` registers the trainer in the active nnU-Net installation.
For an existing nnU-Net environment, see the installation guide.

## Data and pretrained checkpoints

| Dataset | Input | Best checkpoint |
| --- | --- | --- |
| ACDC (`Dataset003_ACDC`) | MRI | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1) |
| AbdomenCT-1K (`Dataset002_AbdomenCT1K`) | CT | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1) |
| AMOS2022 (`Dataset004_AMOS`) | CT only | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1) |

Datasets: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html),
[AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) and
[AMOS2022](https://zenodo.org/records/7262581). See
[docs/DATASETS.md](docs/DATASETS.md) for how to arrange the images.

`predict.sh` downloads a checkpoint, checks its SHA256 and runs
`nnUNetv2_predict`:

```bash
export nnUNet_results="/path/to/nnUNet_results"
bash predict.sh amos-ct /path/to/imagesTs /path/to/predictions
# Replace amos-ct with acdc or abdomenct1k for the other checkpoints.
```

Input files need nnU-Net names (for example `case_0000.nii.gz`). More details
are in [docs/INFERENCE.md](docs/INFERENCE.md).

## Train on your data

Replace `2` with your dataset ID:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
export nnUNet_compile=false
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_SlimFormerPlusPlus
```

This trains on your data. Reproducing an experiment's metrics also requires
the same dataset split, plans, batch size and evaluation settings; the CUDA
smoke test checks execution, not final accuracy.

## Attention variant

[`nnUNetTrainer_SlimFormerPlusPlus_Attention`](nnUNetTrainer_SlimFormerPlusPlus_Attention.py)
uses window self-attention (3 heads, 8×8×8 windows) instead of the selective
mixer in encoder 1. Everything else is the same. It does not need `mamba-ssm`,
`causal-conv1d` or `transformers`. There are no pretrained checkpoints for it.

```bash
# In a fresh Python 3.12 environment:
python -m pip install -c constraints.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c constraints.txt pip setuptools wheel packaging ninja
python check_environment.py --attention --cuda
python -m pip install --no-build-isolation -r requirements-attention.txt
python -m pip check
python install.py --attention
python smoke_test.py --attention --cuda
```

To train it, use `-tr nnUNetTrainer_SlimFormerPlusPlus_Attention`.

## License

Noncommercial research use only. See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md)
and [docs/VALIDATION.md](docs/VALIDATION.md).
