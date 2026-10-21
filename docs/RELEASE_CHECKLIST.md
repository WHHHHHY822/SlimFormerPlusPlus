# Source release checklist

Updated: 2026-10-21.

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
- Full training reproduction and full-dataset prediction with the public runtime.

## Reproduction release

- Three original fold-0 splits, plans, protocols and per-case reference metrics.
- All three released best checkpoints match the original completed runs.
- Reproduction and training entry point, with synthetic evaluation tests.
- Fresh GH200 patch benchmarks, raw timings and a reusable benchmark script.
- No datasets, labels or energy measurements are included.

Publication workflow: [PUBLISHING.md](PUBLISHING.md).
