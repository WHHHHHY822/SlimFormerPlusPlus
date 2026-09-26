# SlimFormer++

3D medical image segmentation with window Mamba and augmented bottleneck mixing,
built for nnU-Net v2. **Source only; pretrained weights are not yet available.**

## Performance

![Dice versus GFLOPs, with parameter counts and pruning comparisons](assets/performance_efficiency_pruning.png)

In the comparison shown, SlimFormer++ achieves approximately **83.4% Dice**
with **less than 200 GFLOPs**, offering the highest Dice and lowest FLOPs among
the displayed methods.

## Efficiency

![Normalized comparison of parameters, FLOPs, peak memory, latency and energy](assets/efficiency_radar.png)

The normalized comparison highlights the lowest parameter count, FLOPs and
peak memory among the compared methods, together with near-best energy
consumption and competitive latency. Farther outward indicates better efficiency.

## Code

The complete model and trainer are in
[`nnUNetTrainer_SlimFormerPlusPlus.py`](nnUNetTrainer_SlimFormerPlusPlus.py):

- `SlimFormerPlusPlus`: encoder, bottleneck, decoder and segmentation head.
- `SelectiveMixer` and `AugmentedMixer`: the two mixing modules.
- `nnUNetTrainer_SlimFormerPlusPlus`: nnU-Net training integration.

## Install

Download this repository and enter its root directory. Use a Python 3.12
environment with CUDA-enabled PyTorch installed first. New installations target
nnunetv2 2.6.2; Mamba requires compatible CUDA extensions.

```bash
python -m pip install setuptools wheel packaging ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python install.py
python smoke_test.py --cuda
```

**Already have nnUNetv2?** Use `requirements-model.txt` instead of
`requirements.txt` to install the model dependencies. The installer registers
the trainer in the active environment and refuses to overwrite a different copy.

## Train

Prepare your dataset in nnU-Net format. Replace the paths and dataset ID below:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
export nnUNet_compile=false

# Skip if already preprocessed.
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_SlimFormerPlusPlus
```

For multiple GPUs, add `-num_gpus N` and set `SLIMFORMER_GLOBAL_BATCH` to a
positive integer divisible by N that fits GPU memory. Add `--c` to resume with
the same configuration. Training uses 1200 epochs, SGD and no deep supervision.

## Predict

After training produces `checkpoint_best.pth`:

```bash
nnUNetv2_predict -i /path/to/imagesTs -o /path/to/predictions \
  -d 2 -c 3d_fullres -tr nnUNetTrainer_SlimFormerPlusPlus \
  -f 0 -chk checkpoint_best.pth
```

## License and validation

**Noncommercial research only.** See [LICENSE](LICENSE) and
[third-party notices](NOTICE.md). Copyright (c) 2026 WHHHHHY822.
Installation and model construction checks passed; full GPU training validation
is pending. See [validation details](docs/VALIDATION.md) and
[contribution instructions](CONTRIBUTING.md).
