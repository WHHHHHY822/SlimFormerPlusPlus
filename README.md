<div align="center">

# SlimFormer++

[![arXiv][arxiv-shield]][arxiv-url]
[![Hugging Face][hf-shield]][hf-paper-url]
[![License][license-shield]](LICENSE)

**Token Mixing Where It Matters for Efficient 3D Medical Image Segmentation**

[Install](docs/INSTALLATION.md) · [Usage](docs/USAGE.md) · [Reproduce](experiments/README.md) · [Validation](docs/VALIDATION.md)

<img src="assets/performance_efficiency_pruning.png" alt="Mean Dice versus GFLOPs, model sizes and pruning comparisons" width="80%" />

**5.98M parameters · 87.84% mean Dice · 6.50 mm mean HD95**<br />
<sub>Paper, Table III: comparison with 12 baselines across eight CT, MRI and ultrasound datasets.</sub>

</div>

SlimFormer++ is an [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) trainer with
windowed Mamba mixing in the first encoder stage, channel MLPs in deeper stages,
and sparse long-range context at the bottleneck. A
[Mamba-free variant](docs/INSTALLATION.md#mamba-free-implementation) is also
available, with no pretrained weights.

## Pretrained models

| Dataset | Input | Dice (%) ↑ | HD95 (mm) ↓ | `predict.sh` key | Weights |
| --- | --- | ---: | ---: | --- | --- |
| ACDC | MRI | 91.56 | 3.70 | `acdc` | [Hugging Face][hf-acdc] · [GitHub][release-acdc] |
| AbdomenCT-1K | CT | 95.62 | 2.84 | `abdomenct1k` | [Hugging Face][hf-abdomenct1k] · [GitHub][release-abdomenct1k] |
| AMOS2022 | CT only | 89.76 | 11.36 | `amos-ct` | [Hugging Face][hf-amos] · [GitHub][release-amos] |

Paper Table III results; the [fold-0 validation records](experiments/README.md)
are separate measurements, not independent test-set results.

## Quick start

Requires Linux, an NVIDIA GPU, Python 3.12 and CUDA Toolkit 12.6 with `nvcc`.
Validated on GH200 and RTX 3090. See [installation](docs/INSTALLATION.md) for
prerequisites, troubleshooting and the Mamba-free variant.

```bash
git clone https://github.com/WHHHHHY822/SlimFormerPlusPlus.git
cd SlimFormerPlusPlus
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -c tools/constraints.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c tools/constraints.txt pip setuptools wheel packaging ninja
python tools/check_environment.py --cuda
MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python tools/install.py
python tools/smoke_test.py --cuda
```

Predict on [prepared](docs/USAGE.md#prepare-data) `*_0000.nii.gz` volumes;
the helper downloads and verifies the selected checkpoint:

```bash
export nnUNet_results=/path/to/checkpoints
bash predict.sh acdc /path/to/imagesTs /path/to/predictions
```

For [training](docs/USAGE.md#train-on-your-data), follow the usage guide.
For evaluation and benchmarks, follow the [reproduction guide](experiments/README.md).

## Citation and license

Use the [BibTeX citation](CITATION.bib) and also cite [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
**Noncommercial research use only**; see [LICENSE](LICENSE) and [third-party notices](NOTICE.md).
This is a source-available research license, not an OSI-approved open-source license.

Built on nnU-Net, [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt),
[Mamba](https://github.com/state-spaces/mamba) and [MONAI](https://github.com/Project-MONAI/MONAI).
We thank the dataset providers listed in the [data guide](docs/USAGE.md#prepare-data).

[arxiv-url]: https://arxiv.org/abs/2610.25488
[hf-paper-url]: https://huggingface.co/papers/2610.25488
[hf-acdc]: https://huggingface.co/Yang0822/SlimFormerPlusPlus-ACDC
[hf-abdomenct1k]: https://huggingface.co/Yang0822/SlimFormerPlusPlus-AbdomenCT1K
[hf-amos]: https://huggingface.co/Yang0822/SlimFormerPlusPlus-AMOS2022-CT
[arxiv-shield]: https://img.shields.io/badge/arXiv-2610.25488-b31b1b.svg?style=flat-square
[hf-shield]: https://img.shields.io/badge/Hugging%20Face-Paper%20%26%20Models-FFD21E.svg?style=flat-square&logo=huggingface
[license-shield]: https://img.shields.io/badge/license-noncommercial%20research-orange.svg?style=flat-square
[release-acdc]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1
[release-abdomenct1k]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1
[release-amos]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1
