# Source release checklist

Updated: 2026-10-06.

## Completed

- Source-only scope; weights and datasets are excluded.
- Copyright attribution: WHHHHHY822.
- Noncommercial research use by any individual or organization is permitted;
  commercial use of covered contributions is prohibited by LICENSE.
- Upstream Apache-2.0 text and attribution are preserved in
  [licenses/Apache-2.0.txt](../licenses/Apache-2.0.txt).
- Model source remains byte-identical to the inspected training version.
- Trainer installer, smoke-check entry point, installer tests and GitHub CI
  are included.
- The concise English README covers installation, training and prediction.
- Fresh dependency installation, trainer discovery and CUDA forward/backward
  with official nnunetv2 2.6.2 passed on GH200;
  see [VALIDATION.md](VALIDATION.md) for the precise scope and limitations.

## Remaining validation

- Linux x86_64 installation and GPU execution.
- Dataset-specific training/prediction, final metrics and multi-GPU equivalence.
- Weight export, evaluation and checksums before any future weight release.

Publication workflow: [PUBLISHING.md](PUBLISHING.md).
