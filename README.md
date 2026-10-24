<div align="center">

# SlimFormer++

[![arXiv][arxiv-shield]][arxiv-url]
[![Hugging Face][hf-shield]][hf-paper-url]
[![3D Slicer extension][slicer-shield]][slicer-url]
[![License][license-shield]](LICENSE)

**Token Mixing Where It Matters for Efficient 3D Medical Image Segmentation**

[Paper][arxiv-url] · [Pretrained weights][releases-url] · [Installation](docs/INSTALLATION.md) · [Usage](docs/USAGE.md) · [Reproduction](experiments/README.md)

<img src="assets/performance_efficiency_pruning.png" alt="Mean Dice versus GFLOPs, with model parameter counts and pruning comparisons" width="53%" />
<img src="assets/efficiency_radar.png" alt="Normalized comparison of parameters, GFLOPs, peak memory, latency and energy; outward is better" width="44%" />

**5.98M** parameters &nbsp;·&nbsp; **87.84%** mean Dice &nbsp;·&nbsp; **6.50 mm** mean HD95<br />
<sub>Highest mean Dice and lowest mean HD95 with the fewest parameters, against 12 baselines on
eight public CT, MRI and ultrasound datasets (paper, Table III).</sub>

</div>

## Overview

SlimFormer++ is an [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) trainer that
keeps token mixing only in the stages where it contributes.

- **Selective Mixing** — windowed Mamba mixing in the first encoder stage;
  deeper stages use residual channel MLPs.
- **Augmented Mixing** — a depthwise convolution at the bottleneck, augmented with
  similarity-weighted sparse long-range context at cost linear in voxel count.
- **Mamba-free implementation** — window attention instead of Mamba, without
  building mamba-ssm or causal-conv1d ([setup](docs/INSTALLATION.md#mamba-free-implementation);
  no pretrained weights).

## Pretrained models

| Dataset | Modality | Dice (%) ↑ | HD95 (mm) ↓ | `predict.sh` key | Weights |
| --- | --- | ---: | ---: | --- | --- |
| ACDC | MRI | 91.56 | 3.70 | `acdc` | [GitHub][release-acdc] · [Hugging Face][hf-acdc] |
| AbdomenCT-1K | CT | 95.62 | 2.84 | `abdomenct1k` | [GitHub][release-abdomenct1k] · [Hugging Face][hf-abdomenct1k] |
| AMOS2022 | CT only | 89.76 | 11.36 | `amos-ct` | [GitHub][release-amos] · [Hugging Face][hf-amos] |

<sub>Dice and HD95 are from Table III of the paper. The fold-0 validation results in
the reproduction bundles are separate measurements, not independent test-set results.</sub>

The [Hugging Face paper page][hf-paper-url] links all three checkpoints to the
paper. Each model page includes its nnU-Net download and prediction instructions.

## Quick start

> [!NOTE]
> Requires Linux, an NVIDIA GPU, Python 3.12, Git, a C++ compiler and CUDA Toolkit
> 12.6 with `nvcc`. Mamba and causal-conv1d compile from source (usually 10–20 min).
> Validated on Linux aarch64 (NVIDIA GH200) and x86_64 (NVIDIA RTX 3090).

**Install**

```bash
git clone https://github.com/WHHHHHY822/SlimFormerPlusPlus.git && cd SlimFormerPlusPlus
python3.12 -m venv .venv && source .venv/bin/activate

# PyTorch and build tools first, then the compiled requirements
python -m pip install -c tools/constraints.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c tools/constraints.txt pip setuptools wheel packaging ninja
python tools/check_environment.py --cuda
MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check

# Register the trainer and run a smoke test
python tools/install.py
python tools/smoke_test.py --cuda
```

**Predict** — downloads and verifies the checkpoint, then runs `nnUNetv2_predict`
on `*_0000.nii.gz` images ([data preparation](docs/USAGE.md#prepare-data)):

```bash
export nnUNet_results=/path/to/checkpoints
bash predict.sh acdc /path/to/imagesTs /path/to/predictions   # or abdomenct1k, amos-ct
```

**Train** — on any nnU-Net dataset; replace `2` with your dataset ID
([environment variables](docs/USAGE.md#train-on-your-data)):

```bash
export nnUNet_compile=false
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_SlimFormerPlusPlus
```

**Reproduce** — evaluates a released checkpoint on the published validation cases
and compares the result with the reference metrics:

```bash
python reproduce.py acdc --images /path/to/images --labels /path/to/labels --output /path/to/evaluation
```

See [Validation](docs/VALIDATION.md) for tested environments and remaining limitations.

## Citation

Please cite SlimFormer++ together with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet):

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

## License and acknowledgments

**Noncommercial research use only.** This is a source-available license, not an
OSI-approved open-source license; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet),
[MedNeXt](https://github.com/MIC-DKFZ/MedNeXt),
[Mamba](https://github.com/state-spaces/mamba) and
[MONAI](https://github.com/Project-MONAI/MONAI). We thank the providers of
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/),
[AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) and
[AMOS2022](https://amos22.grand-challenge.org/).
README layout adapted from [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

[arxiv-url]: https://arxiv.org/abs/2610.25488
[hf-paper-url]: https://huggingface.co/papers/2610.25488
[hf-acdc]: https://huggingface.co/Yang0822/SlimFormerPlusPlus-ACDC
[hf-abdomenct1k]: https://huggingface.co/Yang0822/SlimFormerPlusPlus-AbdomenCT1K
[hf-amos]: https://huggingface.co/Yang0822/SlimFormerPlusPlus-AMOS2022-CT
[arxiv-shield]: https://img.shields.io/badge/arXiv-2610.25488-b31b1b.svg?style=flat-square
[hf-shield]: https://img.shields.io/badge/Hugging%20Face-Paper%20%26%20Models-FFD21E.svg?style=flat-square&logo=huggingface
[slicer-shield]: https://img.shields.io/badge/3D%20Slicer-Extension-1F65B0.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAJwElEQVRYhZ1XCXAUVRr%2B%2Bpie6ZnJXJkr18wkARIIkmC8UECOBUoUKajSBXZrtVzx2ELF3S20XF2VWs9V8VqL0tKqVUt3dZcCsZQgCiosIokiIBIM5JrJZDL32TPT19brBBRMli3%2FqlfT3dPv%2Fd%2F%2F%2Fd%2F7399Uoz%2BAiYyiKKiqikR8BIVcod7l8SyZ1nrJvNra%2BmlfH%2Fw8wPI8SvnU7qPd6RcUiRNAqRRU0FYb1BoXOktlCBMuPmYTAmAYBslEFOlU5rLpbe2%2Fn7tw%2BXWt7VfgwkuvxD%2F%2F%2Fhy%2B7PoC7XOXoXPvp5iMf2PqDAvkHAXGCnxxgELHAcywW3HkfADYnyAai3qg9yTnrqp6%2FqbfPXDrwqXXodLlhdGkx4db3sL2bW9jztJVKJVKKFNWrFtJoWW1DAQZoAGwPwi83UEpduv53J8DgKJoSFIZoWBw%2BqzZC7auv%2B%2BZxsamCxAO9aP%2FVDcqnR58%2BvF2VAWmwGSuwGB%2FP44d3I3IQgNaemmUwoAeQCxNFoN6fvc%2FAkAil2URwWBw1rIVaz6%2Fd%2BNmRpIl9Bw%2FDIqmYXe4cLhzL6LxKGbOXYrIUAjp4R60X9wKtzUMeGXoZRqoAWwmGlCpbknRgIBAYWiAHr2ciAEKwcHBSVctu37fnx57lUqn4kjGR8AwrDbNaK5A93dfQ2VYWOwuHNi9A7%2B58XpU%2Blqw9bXXkc3ZIGZ10DsVHOnmAJZ%2BPFZShikKlKqCc%2BjpKM%2FgFVkdBwARXHDgFNraL92zYeNLVDadRDIeHXM%2Bmpp8Jo2RSBh2dzXCA72wVXCYv3gx9nd8hAdiM%2FFA7zSwQhlSnEdtrgcdC0J%2FNLusKEsAp6Pw6OE89gyLr3h5%2BmwAhPpcNk1yunndhidqOJ0ekaGBM86JcZwBkfAAsvksapra0H2oC02T6uAL%2BLAzloJuphvGZY1gY3nEXVaYCzEsjg8B9QZAVAEDDe8JYTBPrvmzGaAJgGg0Nnn59WtvbW2fjdDgqbOcE%2BONJg2UUCrCandByKU05xo7NA02VwQXz4JL5aGPZ0EJZYwUAeRkCDkZyMoQZFVlzg5%2BlAESfW2d75FF16zS9v14puM4pJJxqKAgiRI4jkaNz4d8QQStAoLNiGK1FZyORcltQ9GshzurAkYavMYACz0lS3IxC3DsadUDOiPYZDxuX7F65XUNk6ZhsL9H2w3nGk0zSKcToBkWuUwKRt4Ap8eLYrEIidfB%2Bf73qEwqoEsSBF4P04kwnilRsAgFiLIKHQP0CUyVw%2BHZCm5sfVXlIZd2snqDfsWM9iugKMqEe5X8J0kidHoDcpkkjCYeFqsNRVlEzKDDX7%2Fsxy%2B700iyLJyigufNOkSmVcBPAQIFWIQYGNtMXrps%2FXJVzQOsASgmQO171Mx6quvm%2B%2BubkMkkx42ePJPEEgqFPPR6I4R8HgaDHkaTGbIsQ5YVOCqN4KscMIACpVCwKWXU22gsMDPISSrsJRpvxjgInBNg%2BFEAo6sH6era%2BhZSZktCYdzoRwGUURTyYHScxgTH6cCwrFZViK4EVaMUOVXVfksqUJRV5KWxIVOQFQWUJADi2JCKWn1h7Q5XDVF5sZCfMAUkUrI2wzJQCgJYVqc95wgTFgsS0SQKxTxSoMAoFIZlEWWzAUkzg6ykAqUsBqL9KLtECBUUkskY2HIaTkUWWIZhTFq9pMapk6Mc%2FKDasRSpqgKT2QxKUfHVjg%2FhuvxyUJMaIZIdAwqKqiBlorGLoUAqn04VscTsR136E3xzsAuXXTQHJSGDPj1Xw6qnnZ%2FPxuglP4Qxs9WKW1atxvR6Hx58601tMkkHkbJpnKXS6RHcu%2FZXuGHZIqy5cR0iwyHccdvOy1lJLGehqOPNGTNVK8UEpCzJMFVYUCqL2LTxISxbtAC333HHhDOT8Qi%2B%2FqoL%2B%2FcfwMGub9HcugJL19wMSc9h2%2Ftb0d%2FbTbPJRHSoUMh6iaiUcnkc9yT3Oi3vJL8ubw1effIetLdOxfKnN6Hjg%2FdQKhUhihIKhQJisQQymSziiQySqSxKig6Vbh9uvvspXHhJAIc7e%2FHco08hm07ggrZZITYc7DsSjw5f6K32QRwPgKKA4zjY7E5E%2Bk4iFjyJW%2B%2B8H54qH1554wPIkqTBVFUKvNEMPW9GhcWL5ovmYcrUGfAHXMikyziwbxe2v7MZoYFTcHursfqmu7HpL%2BuPspGhgY97e47dMKm5FelUYtxaoDcYte3XseV13Hb7PVh710YoioqlpHiR9zVtjBarfC6DeDSM0MBJ7Nj6GmKRIeTzWY1Bnucxd%2FG1mLtkJfZ88C6OH%2Bn6hBVFcduhzr3qoqtXUeM5J5bNptA0tQ2Lf3E1VAp47P5boMgy9Ab%2BDGBJFLVuimF0GhiapjWxOt1ezGpaAk%2BtD6KqYCjUqx14Rzr3oVwuv0vVuF2k1r%2Fx2Iv%2F%2BrWvfgpiI0OjovuRkcisNqfWhg0FexGPDqNUFKAoslYjCAjSU5AoK6x2LV0WqwMGowlQVOSyKXzbfQiCkIfLXQ2IEh6%2Fb%2B2HhUJuKeN1exEKDnZxHHf3%2FMUrtDSca%2BQwEsukHOdA%2BgWH0wO3t1YbRDueqjq4PDVwOL0w8CatDhSFAkhjI4siwpEgMtkUaIpG4%2BQW7NzyJvbu3rHS5fGOMHarjdT29LffHLA0T2%2BfNfWCi7RWjFD4U0WqZw4mAoiM8tjQ7sUyZFnSumoy9HqDViHDI0FtTqChGeHek3j52T%2B%2FXmG1vkx2F%2BkewWsHi%2FSHF5%2Fc0JOIjWgRkYV%2BrhHnOh2npSgcGdTSRRgycjxefeHhSKGQu8loqtAC0sIkgvJW%2B9Fz%2FNi8R%2B77rUA6IkLvzwFBnHOcXrseDPUhlUpoa%2FlqG%2FDSE%2Ffg6KHOeVU1fpn4JEb%2FMFFBrT8Q2v%2FZrkvvX78qR1Ttq286Q%2Bf%2FY0SMBgOv0d07cAKxRASTm2eg2lOHTQ%2Ffqeze%2Bd7sGp%2F%2F%2BI%2FXYxw225kbsoDFYh05%2Bk3XP7q%2B%2BHh%2B45TpnpYZl2gAyHF8%2Bp1zjTwj4iS6iSeiODVwAgbeiNa2WUgMh%2FD0Q%2Bu%2B%2F89nu%2BbX%2Bvxd5J0JAWCsybTZHamBvhOb93RsMWXSyStq6xoQaGgaa0KkM90TcTz6QSMjW8ggnU2C0XMIBKaABY2Ptr2NF5%2FY8LfBgZ5r6%2FwNw6c%2F%2B87y978%2BTknDGo3GWvyB%2Brtmz79mTdvFc0z%2BhmbYHC4tz0QrBFAun0Eul9H6xUR0GMcPd5b3frL9ne9PfPdspbOyq8Ji00COZ%2Bf9PCcjk04ilUi6zBbzVXX%2ByXOqavwttkpXwGJ1mLVOKJvO5zKp%2FnCo%2F1jfqe8%2BTyfTHTa7bajC5hgr0xNoCMB%2FAZvInm44tK1yAAAAAElFTkSuQmCC
[license-shield]: https://img.shields.io/badge/license-noncommercial%20research-orange.svg?style=flat-square
[slicer-url]: https://github.com/WHHHHHY822/SlicerSlimFormerPlusPlus
[releases-url]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases
[release-acdc]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/acdc-best-v1
[release-abdomenct1k]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/abdomenct1k-best-v1
[release-amos]: https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/tag/amos2022-ct-best-v1
