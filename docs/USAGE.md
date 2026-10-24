[← Project home](../README.md)

# Prepare data, predict and train

Follow the [installation guide](INSTALLATION.md), then prepare the images below.
For evaluation with reference labels, use the [reproduction guide](../experiments/README.md).

## Prepare data

Each model expects one image channel as `CASE_ID_0000.nii.gz`. Keep datasets in
separate `imagesTs` folders. Prediction needs neither labels nor manual resampling;
follow each dataset provider's access and use terms.

### ACDC (MRI)

Download and extract the [official ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). Its `training/` and `testing/` folders contain patient subfolders. Use individual cardiac-phase volumes (`patient*_frame*.nii.gz`); exclude `_gt` segmentation masks and `_4d` cine volumes. For example, to prepare the official test images:

```bash
SOURCE=/path/to/ACDC/testing
DEST=/path/to/ACDC_imagesTs
mkdir -p "$DEST"
find "$SOURCE" -type f -name 'patient*_frame*.nii.gz' ! -name '*_gt.nii.gz' ! -name '*_4d.nii.gz' -print0 |
  while IFS= read -r -d '' image; do
    name=$(basename "$image" .nii.gz)
    cp "$image" "$DEST/${name}_0000.nii.gz"
  done
```

### AbdomenCT-1K (CT)

Get the complete dataset through the form linked in the [official AbdomenCT-1K repository](https://github.com/JunMa11/AbdomenCT-1K), then extract its three image parts. Copy the CT images (`Case_*_0000.nii.gz`) into one directory; the `Mask` folder is unnecessary for prediction.

```bash
SOURCE=/path/to/extracted/AbdomenCT-1K
DEST=/path/to/AbdomenCT1K_imagesTs
mkdir -p "$DEST"
find "$SOURCE" -type f -path '*/AbdomenCT-1K-ImagePart*/*' -name 'Case_*_0000.nii.gz' -exec cp -t "$DEST" {} +
```

Point `SOURCE` at the parent directory containing `AbdomenCT-1K-ImagePart1`, `AbdomenCT-1K-ImagePart2`, and `AbdomenCT-1K-ImagePart3` (adjust the folder pattern if your extracted archive differs).

### AMOS2022 (CT only)

Download `amos22.zip` and its metadata CSV from [Zenodo](https://zenodo.org/records/7262581), as linked by the [challenge organizers](https://amos22.grand-challenge.org/Instructions/). The dataset includes both CT and MRI. **Select CT cases only** using the provided metadata and image modality, then put their original `amos_*.nii.gz` volumes in a dedicated `CT_SOURCE` directory. Do not use the MRI cases with this checkpoint. Add the nnU-Net channel suffix when copying:

```bash
CT_SOURCE=/path/to/selected/AMOS_CT_images
DEST=/path/to/AMOS_CT_imagesTs
mkdir -p "$DEST"
for image in "$CT_SOURCE"/amos_*.nii.gz; do
  [ -e "$image" ] || continue
  name=$(basename "$image" .nii.gz)
  cp "$image" "$DEST/${name}_0000.nii.gz"
done
```

## Predict with a released checkpoint

Each release ZIP contains a standard nnU-Net model folder. For example, the
AMOS2022 CT archive extracts to:

```text
nnUNet_results/
└── Dataset004_AMOS/
    └── nnUNetTrainer_SlimFormerPlusPlus__nnUNetPlans__3d_fullres/
        ├── plans.json
        ├── dataset.json
        └── fold_0/
            └── checkpoint_best.pth
```

The plans specify spacing and preprocessing. Checkpoints were verified to load
with the public model using nnUNetv2 2.6.2. Use the matching modality:

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

## Train on your data

Arrange your data in nnU-Net format and replace `2` with its dataset ID:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_compile=false
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_SlimFormerPlusPlus
```

For the Mamba-free implementation, use `-tr nnUNetTrainer_SlimFormerPlusPlus_MambaFree`.
To reproduce the published datasets with their original plans, splits and batch
size, follow the [reproduction guide](../experiments/README.md).
