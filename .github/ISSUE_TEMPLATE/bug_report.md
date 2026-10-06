---
name: Bug report
about: Report an installation, training, or inference problem
title: ''
labels: ''
assignees: ''
---

**Problem and expected behavior**

**Minimal reproduction**
Include the exact command, configuration, and the last 80 lines of the installation
or runtime log, including the final error. Do not attach private data.

**Environment**
- OS / CPU architecture:
- Python / PyTorch / CUDA:
- nnunetv2 / Mamba / causal-conv1d / MedNeXt revision:
- GPU and memory:
- Input shape, first stride, and global batch:

**Checks run**
- `python check_environment.py --cuda` (after installing PyTorch):
- `python -m pip check`:
- `python install.py --check`:
- `python smoke_test.py` (or `--cuda`):
