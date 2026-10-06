"""Check the reference environment before installing CUDA extensions."""
import argparse
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention', action='store_true', help='Skip CUDA build-tool checks')
    parser.add_argument('--cuda', action='store_true', help='Require an available CUDA GPU')
    args = parser.parse_args()
    print(f'Python: {sys.version.split()[0]} ({sys.executable})')
    print(f'Platform: {platform.system()} {platform.machine()}')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('The reference installation requires Python 3.12. Create a fresh environment.')
    from importlib import metadata

    try:
        import torch
        import torchvision
    except Exception as error:
        raise RuntimeError(
            f'PyTorch/torchvision import failed: {error}\n'
            'Install the paired versions from README. In NVIDIA containers, check '
            'LD_LIBRARY_PATH for another installation of torch/lib. '
            'See docs/INSTALLATION.md.'
        ) from error
    for name, module, expected in (('torch', torch, '2.6.0'), ('torchvision', torchvision, '0.21.0')):
        print(f'{name}: {module.__version__} ({module.__file__})')
        if module.__version__.split('+')[0] != expected:
            raise RuntimeError(f'The reference profile requires {name}=={expected}. See README.')
    print(f'PyTorch CUDA: {torch.version.cuda}; C++11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')
    if torch.version.cuda is None:
        raise RuntimeError('CPU-only PyTorch is installed. Install the CUDA build from README.')
    if torch.version.cuda != '12.6':
        raise RuntimeError('The reference profile requires CUDA 12.6 PyTorch. Use the cu126 index from README.')
    available = torch.cuda.is_available()
    print(f'GPU: {torch.cuda.get_device_name(0) if available else "not available on this node"}')
    if args.cuda and not available:
        raise RuntimeError('A CUDA GPU is required for --cuda. Run on a GPU node.')
    if shutil.which('git') is None:
        raise RuntimeError('Git is required to install the pinned source dependencies.')

    if not args.attention:
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Use Linux or WSL2 for the reference Mamba installation.')
        from torch.utils.cpp_extension import CUDA_HOME
        nvcc = Path(CUDA_HOME) / 'bin' / 'nvcc' if CUDA_HOME else None
        if nvcc is None or not nvcc.is_file():
            raise RuntimeError(
                'CUDA Toolkit/nvcc is missing. Install CUDA Toolkit 12.6 and set CUDA_HOME. '
                'The NVIDIA driver and PyTorch wheel do not provide nvcc.'
            )
        output = subprocess.check_output([str(nvcc), '--version'], text=True)
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match is None:
            raise RuntimeError(f'Cannot read the CUDA version from {nvcc}.')
        toolkit = tuple(map(int, match.groups()))
        runtime = tuple(map(int, torch.version.cuda.split('.')[:2]))
        print(f'CUDA Toolkit: {toolkit[0]}.{toolkit[1]} ({nvcc})')
        if toolkit != runtime:
            raise RuntimeError('CUDA Toolkit differs from PyTorch CUDA. Use Toolkit 12.6 for this profile.')
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or shutil.which(compiler[0]) is None:
            raise RuntimeError('A C++ compiler is missing. Install g++ or set CXX.')
        version = subprocess.check_output(compiler + ['--version'], text=True).splitlines()[0]
        print(f'C++ compiler: {version}')

    for name in ('nnunetv2', 'mamba-ssm', 'causal-conv1d', 'transformers'):
        try:
            print(f'{name}: {metadata.version(name)}')
        except metadata.PackageNotFoundError:
            pass
    print('PASS environment checks. Run smoke_test.py --cuda after installing the trainer.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        sys.stdout.flush()
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
