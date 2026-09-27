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

Use Python 3.12 with a CUDA-enabled PyTorch installation. From this repository:

```bash
python -m pip install setuptools wheel packaging ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python install.py
python smoke_test.py --cuda
```

If nnUNetv2 is already installed, use `requirements-model.txt` in place of
`requirements.txt`. The installer adds the trainer to the active environment.

## Pretrained checkpoints

| Dataset | Input | Best checkpoint |
| --- | --- | --- |
| ACDC (`Dataset003_ACDC`) | MRI | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1) |
| AbdomenCT-1K (`Dataset002_AbdomenCT1K`) | CT | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1) |
| AMOS2022 (`Dataset004_AMOS`) | CT only | [Download](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1) |

Install the trainer first, then download and extract the ZIP from the chosen
release into `nnUNet_results`. For example, with the AMOS2022 CT checkpoint:

```bash
export nnUNet_results="/path/to/nnUNet_results"
mkdir -p "$nnUNet_results"
curl -fL -o SlimFormerPlusPlus_AMOS2022_CT_best.zip \
  https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/download/amos2022-ct-best-v1/SlimFormerPlusPlus_AMOS2022_CT_best.zip
unzip SlimFormerPlusPlus_AMOS2022_CT_best.zip -d "$nnUNet_results"
(cd "$nnUNet_results" && sha256sum -c SHA256SUMS)
nnUNetv2_predict -i /path/to/AMOS_CT/imagesTs -o /path/to/predictions \
  -d 4 -c 3d_fullres -tr nnUNetTrainer_SlimFormerPlusPlus \
  -f 0 -chk checkpoint_best.pth
```

For ACDC or AbdomenCT-1K, use its release ZIP and dataset ID (`3` or `2`).
Input files must use the matching modality and nnU-Net naming convention
(e.g. `case_0000.nii.gz`).

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
