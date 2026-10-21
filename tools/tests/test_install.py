import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
from tools.install import install


class InstallerTests(unittest.TestCase):
    def test_install_repeat_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / 'source.py', Path(temp) / 'target.py'
            source.write_bytes(b'original trainer\n')
            self.assertEqual(install(source, target), 'Installed')
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(install(source, target), 'Already installed')
            source.write_bytes(b'different trainer\n')
            with self.assertRaises(FileExistsError):
                install(source, target)
            self.assertEqual(target.read_bytes(), b'original trainer\n')

    def test_cli_finds_models_in_repository_root(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / 'nnunetv2'
            destination = package / 'training/nnUNetTrainer'
            destination.mkdir(parents=True)
            (package / '__init__.py').touch()
            for attention in (False, True):
                flags = ['--attention'] if attention else []
                filename = 'nnUNetTrainer_SlimFormerPlusPlus' + ('_Attention' if attention else '') + '.py'
                result = subprocess.run(
                    [sys.executable, str(root / 'tools/install.py')] + flags,
                    env=dict(os.environ, PYTHONPATH=temp), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, universal_newlines=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((destination / filename).read_bytes(), (root / filename).read_bytes())


if __name__ == '__main__':
    unittest.main()
