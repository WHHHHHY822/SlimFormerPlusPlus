# Checkpoint inference

Install the dependencies and trainer from the [README](../README.md), then set
`nnUNet_results` to a writable folder. Each release ZIP contains a standard
nnU-Net model folder. For example, the AMOS2022 CT archive extracts to:

```text
nnUNet_results/
└── Dataset004_AMOS/
    └── nnUNetTrainer_SlimFormerPlusPlus__nnUNetPlans__3d_fullres/
        ├── plans.json
        ├── dataset.json
        └── fold_0/
            └── checkpoint_best.pth
```

These plans specify the image spacing and preprocessing. The checkpoint was
verified to load with the public model using nnUNetv2 2.6.2. Each model expects
one image channel saved as `.nii.gz` with nnU-Net's `_0000` channel suffix.
Use the matching modality:

| Model argument | Dataset ID | Input modality |
| --- | ---: | --- |
| `acdc` | 3 | MRI |
| `abdomenct1k` | 2 | CT |
| `amos-ct` | 4 | CT only |

The helper downloads the selected best checkpoint if it is not already in
`nnUNet_results`, verifies SHA256 checksums and runs `nnUNetv2_predict` with
configuration `3d_fullres`, trainer `nnUNetTrainer_SlimFormerPlusPlus`, fold `0`
and `checkpoint_best.pth`:

```bash
export nnUNet_results="/path/to/nnUNet_results"
bash predict.sh amos-ct /path/to/imagesTs /path/to/predictions
# Or: bash predict.sh acdc /path/to/imagesTs /path/to/predictions
# Or: bash predict.sh abdomenct1k /path/to/imagesTs /path/to/predictions
```

The helper leaves an existing model directory untouched. If it is incomplete,
remove or relocate that directory before running the helper again. Use an empty
`nnUNet_results` directory to avoid mixing these releases with another model
of the same dataset and trainer name. Output images are written to the requested
output directory. The input images and output directory are never included in
the downloaded release.
