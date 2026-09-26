# Checkpoint release TODO

- [x] ACDC (Dataset003_ACDC), fold 0: publish the best checkpoint with nnU-Net plans, dataset metadata, a checksum and reproduction instructions.
- [ ] AbdomenCT-1K (Dataset002_AbdomenCT1K), fold 0: after training and validation complete, verify and publish **only** the best checkpoint.
- [ ] AMOS2022 (Dataset004_AMOS), fold 0: after training and validation complete, verify and publish **only** the best checkpoint.

For each future checkpoint, verify strict loading against the public model,
remove optimizer and private run metadata, test loading with a standard nnU-Net
installation, record the validation scope, and publish as a GitHub Release asset.
