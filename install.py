"""Register the standalone trainer in the active Python environment's nnU-Net."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

FILENAME = 'nnUNetTrainer_SlimFormerPlusPlus.py'


def install(source, destination):
    """Copy once; never overwrite a different trainer without review."""
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            return 'Already installed'
        raise FileExistsError(
            f'Different trainer already exists: {destination}. '
            'Back it up and remove it explicitly before installing this version.'
        )
    # Exclusive creation also protects against a concurrent installation.
    with destination.open('xb') as output, source.open('rb') as input_file:
        shutil.copyfileobj(input_file, output)
    return 'Installed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check destination without writing')
    args = parser.parse_args()
    spec = importlib.util.find_spec('nnunetv2')
    if spec is None or not spec.submodule_search_locations:
        parser.exit(1, 'nnunetv2 is missing from this Python environment. Install it first.\n')
    roots = [Path(p) / 'training' / 'nnUNetTrainer' for p in spec.submodule_search_locations]
    roots = [p for p in roots if p.is_dir()]
    if len(roots) != 1:
        parser.exit(1, 'Cannot identify a unique nnU-Net trainer directory. Check your environment.\n')
    source = Path(__file__).resolve().with_name(FILENAME)
    destination = roots[0] / FILENAME
    print(f'Python: {sys.executable}\nTrainer: {destination}')
    if args.check:
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            parser.exit(1, 'This release is not installed. Run python install.py.\n')
        print('Installed source matches this release.')
        return
    try:
        print(install(source, destination))
    except OSError as error:
        parser.exit(1, f'{error}\nUse a writable environment; do not run this installer with sudo.\n')


if __name__ == '__main__':
    main()
