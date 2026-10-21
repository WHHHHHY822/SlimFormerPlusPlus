# SlimFormer++

SlimFormer++ is a 3D medical image segmentation network for nnU-Net v2.

[Paper](https://arxiv.org/abs/2610.25488) ·
[Model and trainer](nnUNetTrainer_SlimFormerPlusPlus.py) ·
[Documentation](docs/README.md)

<p align="center">
  <img src="assets/performance_efficiency_pruning.png" alt="Segmentation performance versus computational cost" width="53%" />
  <img src="assets/efficiency_radar.png" alt="Normalized efficiency comparison" width="44%" />
</p>

## Installation

Reference environment: **Linux, Python 3.12, PyTorch 2.6.0 and CUDA Toolkit 12.6**.
Follow the [installation guide](docs/INSTALLATION.md) for setup, GPU compatibility,
troubleshooting and the [Mamba-free implementation](docs/INSTALLATION.md#mamba-free-implementation).

## Pretrained checkpoints

| Dataset | Input | Download |
| --- | --- | --- |
| ACDC | MRI | [Best checkpoint + reproduction package](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1) |
| AbdomenCT-1K | CT | [Best checkpoint + reproduction package](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1) |
| AMOS2022 | CT only | [Best checkpoint + reproduction package](https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1) |

After installation, predict with one command:

```bash
export nnUNet_results=/path/to/checkpoints
bash predict.sh acdc /path/to/imagesTs /path/to/predictions
```

Use `acdc`, `abdomenct1k` or `amos-ct`. The helper downloads and verifies the weights.
See [data preparation](docs/DATASETS.md) for image naming and [usage](docs/USAGE.md)
for inference details and training on your own data.

## Reproduce results

Evaluate the released checkpoint on its published fold-0 validation cases:

```bash
python reproduce.py acdc --images /path/to/images --labels /path/to/labels --output /path/to/evaluation
```

The [reproduction guide](experiments/README.md) contains the reference metrics,
training commands and efficiency benchmarks for all three datasets.
Tested environments and verification scope are recorded in [Validation](docs/VALIDATION.md).

## Citation

Please cite [SlimFormer++](https://arxiv.org/abs/2610.25488) ([BibTeX](CITATION.bib))
and nnU-Net when using this project in research.

## License

Noncommercial research use only. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
