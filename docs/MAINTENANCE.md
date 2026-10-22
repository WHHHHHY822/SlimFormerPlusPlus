[← Project home](../README.md) · [Documentation](README.md)

# Release maintenance

## Before publishing

1. Preserve [LICENSE](../LICENSE), [NOTICE.md](../NOTICE.md) and upstream notices.
2. Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md).
3. Update [VALIDATION.md](VALIDATION.md) with completed checks and their limits.
4. Review the staged changes; exclude credentials, images, labels, logs and weights.
5. Commit and push the reviewed changes. Publish weights separately as Release assets.

The project uses a noncommercial research license; describe its terms accurately.

## Checkpoint releases

ACDC, AbdomenCT-1K and AMOS CT best checkpoints and reproduction materials are
published. Their splits, settings, per-case metrics and benchmarks are documented
in the [reproduction guide](../experiments/README.md).

For a new checkpoint, verify strict loading with the public model and nnU-Net
predictor, remove optimizer/private run metadata, and include plans, dataset
metadata, a short protocol, evaluation results and SHA256 checksums.

## Remaining validation

- Linux x86_64 installation and GPU execution.
- Full training reproduction and full-dataset prediction with the public runtime.
- Multi-GPU equivalence.

Earlier planning notes are retained in [the archive](archive/OPTIMIZATION.md).
