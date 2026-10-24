# Contributing and releases

Discuss architecture or training-behavior changes in an issue before opening a
pull request. Use a dedicated environment and follow the
[installation guide](../docs/INSTALLATION.md).

## Checks

```bash
python -m unittest discover -s tools/tests -v
python -m compileall -q tools nnUNetTrainer_SlimFormerPlusPlus.py \
    nnUNetTrainer_SlimFormerPlusPlus_MambaFree.py reproduce.py benchmark.py

# With model dependencies installed:
python tools/check_environment.py
python tools/install.py --check
python tools/smoke_test.py --cuda
```

For the Mamba-free variant, add `--mamba-free` to the three model checks.
Omit `--cuda` from the smoke test on a CPU node. The installer refuses to
overwrite a different trainer; back it up and explicitly remove it before
installing changed model code.

Preserve checkpoint keys and tensor shapes unless a compatibility change is
intentional and documented. Report checks actually run, runtime versions and
unavailable checks. CI covers syntax, installer behavior, experiment consistency
and evaluation on synthetic images; it does not run GPU training.

## Publishing

Preserve [LICENSE](../LICENSE), [NOTICE.md](../NOTICE.md) and third-party notices.
Submit contributions you have the right to share under the noncommercial
research terms. Exclude images, patient identifiers, credentials, logs and
checkpoints from source commits. For bug reports, include a minimal reproduction,
sanitized traceback, dependency/GPU/CUDA versions and input shapes.

Publish checkpoints separately as GitHub Release assets or Hugging Face models.
Verify strict loading with the public model and nnU-Net predictor, remove
optimizer/private run metadata, and include plans, dataset metadata, protocol,
metrics and SHA256 checksums. Update the [validation record](../docs/VALIDATION.md)
with completed checks and their limits before publishing reviewed changes.

Remaining validation includes full training reproduction, full-dataset prediction
with the public runtime and multi-GPU equivalence. Shape/boundary tests and
end-to-end profiling should precede measured model optimizations.

README layout adapted from
[Best-README-Template](https://github.com/othneildrew/Best-README-Template).
