# Checkpoint release TODO

- [x] ACDC (Dataset003_ACDC), fold 0: publish the best checkpoint with nnU-Net plans, dataset metadata, a checksum and reproduction instructions.
- [x] AbdomenCT-1K (Dataset002_AbdomenCT1K), fold 0: publish **only** the best checkpoint.
- [x] AMOS2022 (Dataset004_AMOS), fold 0: publish **only** the best checkpoint.

For each checkpoint, verify strict loading against the public model,
remove optimizer and private run metadata, test loading with a standard nnU-Net
installation, record the validation scope, and publish as a GitHub Release asset.
