# Installation

The [README](../README.md#installation) installs the original SlimFormer++
model. Use its reference profile before adapting package versions.

| Component | Reference profile |
| --- | --- |
| Python | 3.12 |
| PyTorch / torchvision | 2.6.0+cu126 / 0.21.0 |
| CUDA Toolkit | 12.6, with `nvcc` and a compatible C++ compiler |
| nnU-Net / dynamic-network-architectures | 2.6.2 / 0.4.1 |
| Mamba / causal-conv1d | 2.2.2 / 1.4.0, pinned GitHub commits |
| Validated system | Linux aarch64, NVIDIA GH200 |

Linux x86_64 follows the same source-build procedure but has not been validated
in this release. Use Linux through WSL2 on Windows. macOS and CPU-only execution
are outside the original model's reference profile.

## GPU compatibility

| GPU | Current reference profile |
| --- | --- |
| GH200 | Fresh installation and CUDA forward/backward validated. |
| RTX 3090 / 4090 | Expected to work from CUDA architecture compatibility; not tested on these GPUs. |
| RTX 5090 / other Blackwell GPUs | Unsupported by this profile; requires a separately validated environment. |

The RTX 3090 and 4090 have compute capabilities 8.6 and 8.9. The pinned
extensions compile `sm_80` kernels, which are compatible with both under
[NVIDIA's CUDA compatibility rules](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html).
The RTX 5090 has compute capability 12.0; see the
[NVIDIA GPU table](https://developer.nvidia.com/cuda/gpus).
[PyTorch 2.7 introduced Blackwell support with CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/).
The pinned Mamba and causal-conv1d builds also need compatible kernels, so
upgrading PyTorch alone does not make this profile support the RTX 5090.

Full training memory requirements depend on the dataset plans, patch size and
batch size. Passing the CUDA smoke test does not validate full training or
reproduction of published metrics.

## Before installing

- Install a CUDA-capable NVIDIA driver, Git, Python 3.12, a C++ compiler and
  CUDA Toolkit 12.6. A working `nvidia-smi` or CUDA PyTorch installation does
  not provide the `nvcc` compiler.
- Install PyTorch and torchvision together using the README versions and
  CUDA wheel index. Install them before building Mamba.
- `constraints.txt` records the dependency versions from the validated
  installation. Requirements pin the CUDA extensions to immutable commits
  containing their complete CUDA sources.
- Run `python check_environment.py --cuda` before installing the remaining
  requirements. On a cluster's build/login node, omit `--cuda`; run the CUDA
  smoke test later on a GPU node.
- `--no-build-isolation` makes the CUDA extensions build against the PyTorch
  already installed in your environment. `MAX_JOBS=4` limits compiler memory
  use; use `MAX_JOBS=1` if compilation exhausts memory.

The pinned Mamba and causal-conv1d releases have no torch2.6 prebuilt wheels.
Source compilation is expected. Their PyPI source archives omit `csrc` files,
so installing these old versions directly from PyPI can fail when a wheel is
unavailable. Use the repository requirements instead.

## NVIDIA containers

A `devel` image provides compilation tools; a `runtime` image may not.
Check `nvcc --version` before creating the environment.

Some NVIDIA images prepend their bundled PyTorch libraries to
`LD_LIBRARY_PATH`. Installing official PyTorch in a virtual environment can
then mix new Python files with old shared libraries, producing an `XCCL`
attribute error or an undefined-symbol error. In the shell used for the
**new virtual environment**, remove the bundled torch library paths before
following the README:

```bash
export LD_LIBRARY_PATH="$(python -c 'import os; print(":".join(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if "/torch/lib" not in p))')"
```

Use that environment when running training or prediction as well. This is a
shell-local change; it does not modify the container or another environment.

If pip repeatedly retries the obsolete `pypi.ngc.nvidia.com` package source,
disable the image's pip configuration in the same shell:

```bash
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL
```

Both issues were encountered and resolved during the fresh GH200 installation.

## Existing nnU-Net environments

Prefer a new environment to preserve other experiments. If using an existing
environment, first check that its Python, PyTorch, torchvision, nnU-Net and
dynamic-network-architectures versions match the reference profile. Then:

```bash
python -m pip install -c constraints.txt pip setuptools wheel packaging ninja
python check_environment.py --cuda
MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements-model.txt
python -m pip check
python install.py
python smoke_test.py --cuda
```

Run these commands from the repository root. `install.py` refuses to overwrite
a different trainer; back up and explicitly remove that file before replacing
it. To check an existing installation without writing, run
`python install.py --check`.

## Troubleshooting

| Error | Action |
| --- | --- |
| `nvcc` missing / `CUDA_HOME` unset | Install CUDA Toolkit 12.6. Set `CUDA_HOME` to its root and add `$CUDA_HOME/bin` to `PATH`. |
| CUDA version mismatch | Compare `nvcc --version` with `torch.version.cuda`; use Toolkit 12.6 with the reference cu126 PyTorch build. |
| `csrc/... missing and no known rule to make it` | Use the pinned GitHub requirements instead of the old PyPI source archives. |
| `XCCL` / undefined symbol on importing torch | Check for bundled torch libraries in `LD_LIBRARY_PATH`; see the NVIDIA container instructions above. |
| Undefined symbol on importing a CUDA extension | Rebuild the extension against the active PyTorch/CUDA/C++ ABI. Do not reuse wheels from another environment. |
| Compiler killed / out of memory | Retry with `MAX_JOBS=1`. |
| `No module named torch` during a build | Install PyTorch first, then use `--no-build-isolation`. |

For an installation issue, include the exact command, the final 80 log lines,
`python check_environment.py --cuda`, and `python -m pip check` output. The final
compiler error is more useful than a screenshot containing only the traceback.
