import tempfile
import unittest
from pathlib import Path
from install import install


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


if __name__ == '__main__':
    unittest.main()
