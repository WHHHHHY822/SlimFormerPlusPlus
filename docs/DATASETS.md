[← Project home](../README.md) · [Documentation](README.md)

# Dataset download and inference preparation

These steps prepare NIfTI images for [checkpoint prediction](USAGE.md#predict-with-a-released-checkpoint).
Keep each dataset in its own `imagesTs` directory. Labels and manual resampling
are unnecessary for prediction. Follow each dataset's access and use terms.
For evaluation with reference labels, see the [reproduction guide](../experiments/README.md).

## ACDC (MRI)

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

## AbdomenCT-1K (CT)

Get the complete dataset through the form linked in the [official AbdomenCT-1K repository](https://github.com/JunMa11/AbdomenCT-1K), then extract its three image parts. Copy the CT images (`Case_*_0000.nii.gz`) into one directory; the `Mask` folder is unnecessary for prediction.

```bash
SOURCE=/path/to/extracted/AbdomenCT-1K
DEST=/path/to/AbdomenCT1K_imagesTs
mkdir -p "$DEST"
find "$SOURCE" -type f -path '*/AbdomenCT-1K-ImagePart*/*' -name 'Case_*_0000.nii.gz' -exec cp -t "$DEST" {} +
```

Point `SOURCE` at the parent directory containing `AbdomenCT-1K-ImagePart1`, `AbdomenCT-1K-ImagePart2`, and `AbdomenCT-1K-ImagePart3` (adjust the folder pattern if your extracted archive differs).

## AMOS2022 (CT only)

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

Run these commands from the repository root. Once images are arranged, follow
the [prediction command](USAGE.md#predict-with-a-released-checkpoint).
