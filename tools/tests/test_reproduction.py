import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproduce import (DATASETS, METRICS, ROOT, evaluate_case, file_index, find_case,
                       load_bundle, prepare_training, summarize, training_memory_error)

HAS_IMAGES = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'scipy', 'SimpleITK'))


class PublishedExperimentTests(unittest.TestCase):
    def test_reference_cases_metrics_and_source_match(self):
        model_hash = hashlib.sha256((ROOT / 'nnUNetTrainer_SlimFormerPlusPlus.py').read_bytes()).hexdigest()
        for alias in DATASETS:
            with self.subTest(dataset=alias):
                folder, protocol, dataset, split = load_bundle(alias)
                self.assertEqual(protocol['trainer_sha256'], model_hash)
                self.assertEqual(protocol['training']['global_batch'], 8)
                self.assertEqual(protocol['split']['validation_cases'], len(split['val']))
                self.assertEqual(protocol['split']['train_cases'], len(split['train']))
                with (folder / 'metrics.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    for metric in METRICS:
                        row[metric] = float(row[metric]) if row[metric] else None
                expected_labels = {value for value in dataset['labels'].values() if value != 0}
                self.assertEqual({row['case'] for row in rows}, set(split['val']))
                pairs = {(row['case'], int(row['label'])) for row in rows}
                self.assertEqual(len(pairs), len(rows))
                self.assertEqual(pairs, {(case, label) for case in split['val'] for label in expected_labels})
                self.assertEqual(summarize(rows), json.loads((folder / 'metrics.json').read_text()))
                benchmark = json.loads((folder / 'benchmark.json').read_text())
                historical = json.loads((folder / 'efficiency.json').read_text())
                self.assertEqual(len(benchmark['latency_ms']), 50)
                self.assertEqual(benchmark['params'], historical['params'])
                self.assertAlmostEqual(benchmark['gflops_total'], historical['gflops_total'])
                self.assertEqual(benchmark['benchmark_script_sha256'],
                                 hashlib.sha256((ROOT / 'benchmark.py').read_bytes()).hexdigest())
                if alias == 'acdc':
                    patients = lambda cases: {case.rsplit('_frame', 1)[0] for case in cases}
                    self.assertFalse(patients(split['train']) & patients(split['val']))

    def test_macro_aggregation_ignores_empty_case_label(self):
        rows = [dict(case=case, label=label, **{metric: value for metric in METRICS})
                for case, label, value in [('a', 1, 1), ('b', 1, 0), ('a', 2, 1), ('b', 2, None)]]
        self.assertEqual(summarize(rows)['foreground_mean']['Dice'], 0.75)

    def test_native_acdc_names_and_missing_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'patient001_frame01.nii.gz').touch()
            (root / 'patient001_frame01_gt.nii.gz').touch()
            index = file_index(root)
            self.assertEqual(find_case(index, 'ACDC_patient001_frame01').name, 'patient001_frame01.nii.gz')
            self.assertEqual(find_case(index, 'ACDC_patient001_frame01', True).name, 'patient001_frame01_gt.nii.gz')
            # In a labels-only folder, native _gt files are supported.
            labels = {'patient001_frame01_gt.nii.gz': [root / 'patient001_frame01_gt.nii.gz']}
            self.assertEqual(find_case(labels, 'ACDC_patient001_frame01', True).name, 'patient001_frame01_gt.nii.gz')
            with self.assertRaises(FileNotFoundError):
                find_case(index, 'ACDC_patient002_frame01')

    def test_training_setup_preserves_other_experiments(self):
        folder, protocol, dataset, _ = load_bundle('acdc')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / 'images', root / 'labels'
            images.mkdir()
            labels.mkdir()
            for case in ('a', 'b'):
                (images / f'{case}_0000.nii.gz').touch()
                (labels / f'{case}.nii.gz').touch()
            environment = {f'nnUNet_{key}': str(root / key) for key in ('raw', 'preprocessed', 'results')}
            split = {'train': ['a'], 'val': ['b']}
            with patch.dict(os.environ, environment):
                prepare_training(folder, protocol, dataset, split, images, labels)
                self.assertEqual((root / 'preprocessed' / protocol['dataset'] / 'dataset.json').read_bytes(),
                                 (folder / 'dataset.json').read_bytes())
                destination = root / 'preprocessed' / protocol['dataset'] / 'splits_final.json'
                destination.write_text('another experiment')
                with self.assertRaises(FileExistsError):
                    prepare_training(folder, protocol, dataset, split, images, labels)
                self.assertEqual(destination.read_text(), 'another experiment')

    def test_training_memory_check_matches_24gb_limits(self):
        gib = 23.57  # RTX 3090, as reported by PyTorch
        # Measured on 24 GB: per-GPU batch 4 fits ACDC and AbdomenCT-1K, 2 fits AMOS; the next valid batch does not.
        self.assertIn('--gpus 2', training_memory_error('acdc', 8, 1, gib))
        self.assertIsNone(training_memory_error('acdc', 8, 2, gib))
        self.assertIn('--gpus 2', training_memory_error('abdomenct1k', 8, 1, gib))
        self.assertIsNone(training_memory_error('abdomenct1k', 8, 2, gib))
        self.assertIn('--gpus 4', training_memory_error('amos-ct', 8, 2, gib))
        self.assertIsNone(training_memory_error('amos-ct', 8, 4, gib))
        self.assertIsNone(training_memory_error('acdc', 8, 1, 94.5))
        self.assertIn('more memory', training_memory_error('amos-ct', 8, 8, 6.0))


@unittest.skipUnless(HAS_IMAGES, 'Requires numpy, scipy and SimpleITK')
class ImageEvaluationTests(unittest.TestCase):
    def test_anisotropic_distances_empty_masks_and_geometry(self):
        import numpy as np
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, pred = np.zeros((5, 5, 5), dtype=np.uint8), np.zeros((5, 5, 5), dtype=np.uint8)
            ref[1, 2, 2], pred[2, 2, 2], ref[4, 4, 4] = 1, 1, 3
            for name, array in [('ref', ref), ('pred', pred)]:
                image = sitk.GetImageFromArray(array)
                image.SetSpacing((1, 2, 3))
                sitk.WriteImage(image, str(root / f'{name}.nii.gz'))
            task = ('case', root / 'ref.nii.gz', root / 'pred.nii.gz', {1: 'organ', 2: 'empty', 3: 'miss'})
            rows = evaluate_case(task)
            self.assertEqual(rows[0]['HD95_mm'], 3)
            self.assertEqual(rows[0]['ASD_mm'], 3)
            self.assertIsNone(rows[1]['Dice'])
            self.assertAlmostEqual(rows[2]['HD95_mm'], math.sqrt(224))
            self.assertEqual(rows[2]['Dice'], 0)
            image.SetOrigin((1, 0, 0))
            sitk.WriteImage(image, str(root / 'pred.nii.gz'))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                evaluate_case(task)

    def test_cli_uses_exact_validation_cases(self):
        import numpy as np
        import SimpleITK as sitk

        _, _, _, split = load_bundle('acdc')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'labels'
            inputs.mkdir()
            array = np.zeros((6, 6, 6), dtype=np.uint8)
            array[1, 1, 1], array[2, 2, 2], array[3, 3, 3] = 1, 2, 3
            image = sitk.GetImageFromArray(array)
            first = inputs / f'{split["val"][0]}.nii.gz'
            sitk.WriteImage(image, str(first))
            for case in split['val'][1:]:
                (inputs / f'{case}.nii.gz').symlink_to(first)
            (inputs / 'not_in_split.nii.gz').write_bytes(b'invalid unrelated file')
            output = root / 'evaluation'
            completed = subprocess.run(
                [sys.executable, str(ROOT / 'reproduce.py'), 'acdc', '--labels', str(inputs),
                 '--predictions', str(inputs), '--output', str(output), '--workers', '2'],
                capture_output=True, text=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads((output / 'metrics.json').read_text())
            self.assertEqual(result['n_cases'], 60)
            self.assertEqual(result['foreground_mean']['Dice'], 1)
            self.assertEqual(result['foreground_mean']['HD95_mm'], 0)


if __name__ == '__main__':
    unittest.main()
