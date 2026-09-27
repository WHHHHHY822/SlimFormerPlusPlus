# SlimFormer++

SlimFormer++ is a 3D medical image segmentation model for nnU-Net v2.
The [model and trainer](nnUNetTrainer_SlimFormerPlusPlus.py) are provided in one file.

## Performance and efficiency

<p align="center">
  <img src="assets/performance_efficiency_pruning.png" alt="Segmentation performance versus computational cost" width="49%" />
  <img src="assets/efficiency_radar.png" alt="Normalized efficiency comparison" width="49%" />
</p>

SlimFormer++ shows strong segmentation performance at low computational cost.
The efficiency comparison highlights its small model size, low memory use and
competitive runtime among the methods shown.

## Installation

Use Python 3.12. First install PyTorch **and** torchvision built for your CUDA
driver (choose the matching `cu*` index on [pytorch.org](https://pytorch.org/get-started/previous-versions/)).
Install both together: if torchvision is missing, the dependencies below pull
the newest torchvision, and pip replaces your PyTorch with the version it needs.
Then, from this repository:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install setuptools wheel packaging ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python install.py
python smoke_test.py --cuda
```

`mamba-ssm` and `causal-conv1d` are installed from their release tags. Prebuilt
wheels exist only for x86_64 with PyTorch 2.0–2.4; otherwise they compile from
source, which needs the CUDA toolkit (`nvcc`) matching PyTorch's CUDA version
and can take tens of minutes. Set `MAX_JOBS=4` if memory is limited.

If nnUNetv2 is already installed, use `requirements-model.txt` in place of
`requirements.txt`. The installer adds the trainer to the active environment.
Tested with Python 3.12.3, PyTorch 2.6.0 (CUDA 12.6), torchvision 0.21.0 and
nnUNetv2 2.6.2 on Linux aarch64 (NVIDIA GH200).

## Data and pretrained checkpoints

| Dataset | Input | Best checkpoint |
| --- | --- | --- |
| ACDC (`Dataset003_ACDC`) | MRI | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1) |
| AbdomenCT-1K (`Dataset002_AbdomenCT1K`) | CT | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1) |
| AMOS2022 (`Dataset004_AMOS`) | CT only | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1) |

Download the datasets from their official sources: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html),
[AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K), and
[AMOS2022](https://zenodo.org/records/7262581). Follow the
[data preparation guide](docs/DATASETS.md) to arrange images for inference.

Install the trainer, set `nnUNet_results`, then run inference with the
matching input modality. The helper downloads the selected best checkpoint,
verifies it and applies the packaged nnU-Net plans:

```bash
export nnUNet_results="/path/to/nnUNet_results"
bash predict.sh amos-ct /path/to/imagesTs /path/to/predictions
# Replace amos-ct with acdc or abdomenct1k for the other checkpoints.
```

Inputs must use nnU-Net naming (for example `case_0000.nii.gz`). See the
[full inference configuration](docs/INFERENCE.md) for the archive layout,
modalities and command arguments.

## Train on your data

Prepare an nnU-Net dataset and set `nnUNet_raw`, `nnUNet_preprocessed` and
`nnUNet_results`. Replace `2` with your dataset ID:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
export nnUNet_compile=false
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_SlimFormerPlusPlus
```

## License

Noncommercial research use only. See [LICENSE](LICENSE), [third-party
notices](NOTICE.md) and [validation notes](docs/VALIDATION.md).
