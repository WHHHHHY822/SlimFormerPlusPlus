<a id="readme-top"></a>

[![arXiv][arxiv-shield]][arxiv-url]
[![Source checks][checks-shield]][checks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

<br />
<div align="center">
  <h1 align="center">SlimFormer++</h1>

  <p align="center">
    Token mixing where it matters for efficient 3D medical image segmentation, as an nnU-Net v2 trainer.
    <br />
    <a href="docs/README.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://arxiv.org/abs/2610.25488">Paper</a>
    &middot;
    <a href="https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases">Pretrained Weights</a>
    &middot;
    <a href="https://github.com/WHHHHHY822/SlimFormerPlusPlus/issues/new?labels=bug&template=bug_report.md">Report Bug</a>
    &middot;
    <a href="https://github.com/WHHHHHY822/SlimFormerPlusPlus/issues/new?labels=enhancement&template=feature_request.md">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

<p align="center">
  <img src="assets/performance_efficiency_pruning.png" alt="Segmentation performance versus computational cost" width="53%" />
  <img src="assets/efficiency_radar.png" alt="Normalized efficiency comparison" width="44%" />
</p>

SlimFormer++ keeps token mixing only in the stages where it contributes:

* **Selective Mixing** uses windowed Mamba mixing in the first encoder stage and
  only residual channel MLPs in deeper stages.
* **Augmented Mixing** at the bottleneck adds similarity-weighted sparse
  long-range context to a depthwise convolution, at cost linear in voxel count.

Against 12 baselines on eight public CT, MRI and ultrasound datasets, it achieved
the highest mean Dice (87.84%) and lowest mean HD95 (6.50 mm) with the fewest
parameters (5.98M) ([paper](https://arxiv.org/abs/2610.25488)). A
[Mamba-free implementation](docs/INSTALLATION.md#mamba-free-implementation)
is also provided.

[![Python][Python-shield]][Python-url]
[![PyTorch][PyTorch-shield]][PyTorch-url]
[![nnU-Net][nnUNet-shield]][nnUNet-url]
[![Mamba][Mamba-shield]][Mamba-url]
[![MedNeXt][MedNeXt-shield]][MedNeXt-url]
[![MONAI][MONAI-shield]][MONAI-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

* Linux with an NVIDIA GPU, Python 3.12, Git and a C++ compiler
* CUDA Toolkit 12.6 including `nvcc`

Validated on Linux aarch64 with an NVIDIA GH200. See the
[installation guide](docs/INSTALLATION.md) for GPU compatibility, NVIDIA
containers and troubleshooting.

### Installation

Mamba and causal-conv1d are compiled from source (usually 10–20 minutes).

1. Clone the repository and create an environment
   ```sh
   git clone https://github.com/WHHHHHY822/SlimFormerPlusPlus.git
   cd SlimFormerPlusPlus
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
2. Install PyTorch, torchvision and the build tools first
   ```sh
   python -m pip install -c tools/constraints.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
   python -m pip install -c tools/constraints.txt pip setuptools wheel packaging ninja
   ```
3. Check the environment, then build the remaining requirements
   ```sh
   python tools/check_environment.py --cuda
   MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements.txt
   python -m pip check
   ```
4. Register the trainer with nnU-Net and run the smoke test
   ```sh
   python tools/install.py
   python tools/smoke_test.py --cuda
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

| Dataset | Model argument | Input | Dice (%) | HD95 (mm) | Checkpoint |
| --- | --- | --- | ---: | ---: | --- |
| ACDC | `acdc` | MRI | 91.60 | 3.30 | [acdc-best-v1][release-acdc] |
| AbdomenCT-1K | `abdomenct1k` | CT | 95.49 | 2.89 | [abdomenct1k-best-v1][release-abdomenct1k] |
| AMOS2022 | `amos-ct` | CT only | 90.40 | 10.01 | [amos2022-ct-best-v1][release-amos] |

Metrics are on the published fold-0 validation cases, not an independent test set.

**Predict.** The helper downloads and verifies the checkpoint, then runs
`nnUNetv2_predict` on `_0000.nii.gz` images ([data preparation](docs/DATASETS.md)):

```sh
export nnUNet_results=/path/to/checkpoints
bash predict.sh acdc /path/to/imagesTs /path/to/predictions
```

**Train on your data.** With data in nnU-Net format (replace `2` with your dataset ID):

```sh
export nnUNet_compile=false
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_SlimFormerPlusPlus
```

**Reproduce the released results.**

```sh
python reproduce.py acdc --images /path/to/images --labels /path/to/labels --output /path/to/evaluation
```

See the [usage guide](docs/USAGE.md) for environment variables and the Mamba-free
trainer, and the [reproduction guide](experiments/README.md) for splits,
per-case metrics, training from scratch and efficiency benchmarks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Pretrained checkpoints for ACDC, AbdomenCT-1K and AMOS2022 CT
- [x] Reproduction bundles and GPU benchmarks
- [x] Mamba-free implementation (no pretrained checkpoint)
- [ ] Validate installation and GPU execution on Linux x86_64
- [ ] Full training reproduction and full-dataset prediction with the public runtime
- [ ] Multi-GPU equivalence checks

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Please open an issue to discuss architecture or training changes before
submitting a pull request, and run the checks in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Noncommercial research use only. This is a source-available license, not an
OSI-approved open-source license. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

## Citation

Please cite SlimFormer++ and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet):

```bibtex
@misc{hong2026slimformertokenmixingmatters,
      title={SlimFormer++: Token Mixing Where It Matters for Efficient 3D Medical Image Segmentation},
      author={Yang Hong and Lei Zhang and Zhihong Liu and Jianqing Mo and Xujiong Ye},
      year={2026},
      eprint={2610.25488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.25488},
}
```

## Acknowledgments

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet),
[MedNeXt](https://github.com/MIC-DKFZ/MedNeXt),
[Mamba](https://github.com/state-spaces/mamba) and
[MONAI](https://github.com/Project-MONAI/MONAI). We thank the
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/),
[AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) and
[AMOS2022](https://amos22.grand-challenge.org/) dataset providers. README based on
[Best-README-Template](https://github.com/othneildrew/Best-README-Template).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[arxiv-shield]: https://img.shields.io/badge/arXiv-2610.25488-b31b1b.svg?style=for-the-badge
[arxiv-url]: https://arxiv.org/abs/2610.25488
[checks-shield]: https://img.shields.io/github/actions/workflow/status/WHHHHHY822/SlimFormerPlusPlus/checks.yml?branch=main&style=for-the-badge&label=source%20checks
[checks-url]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/actions/workflows/checks.yml
[stars-shield]: https://img.shields.io/github/stars/WHHHHHY822/SlimFormerPlusPlus.svg?style=for-the-badge
[stars-url]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/stargazers
[issues-shield]: https://img.shields.io/github/issues/WHHHHHY822/SlimFormerPlusPlus.svg?style=for-the-badge
[issues-url]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/issues
[license-shield]: https://img.shields.io/badge/license-noncommercial%20research-orange.svg?style=for-the-badge
[license-url]: LICENSE
[release-acdc]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1
[release-abdomenct1k]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1
[release-amos]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1
[Python-shield]: https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[nnUNet-shield]: https://img.shields.io/badge/nnU--Net-v2-1F6FEB?style=for-the-badge
[nnUNet-url]: https://github.com/MIC-DKFZ/nnUNet
[Mamba-shield]: https://img.shields.io/badge/Mamba-2.2.2-6E40C9?style=for-the-badge
[Mamba-url]: https://github.com/state-spaces/mamba
[MedNeXt-shield]: https://img.shields.io/badge/MedNeXt-v1-2E7D32?style=for-the-badge
[MedNeXt-url]: https://github.com/MIC-DKFZ/MedNeXt
[MONAI-shield]: https://img.shields.io/badge/MONAI-1.4-00A3A3?style=for-the-badge
[MONAI-url]: https://monai.io/
