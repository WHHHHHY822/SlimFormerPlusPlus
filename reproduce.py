"""Reproduce a released fold-0 result, or train with its published split and plans."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
DATASETS = ('acdc', 'abdomenct1k', 'amos-ct')
METRICS = ('Dice', 'IoU', 'HD95_mm', 'ASD_mm')
# Training memory per sample on one GPU (GiB): peak reserved at per-GPU batch 1,
# FP16 autocast, RTX 3090. The overhead covers the CUDA context and loss.
TRAINING_GIB_PER_SAMPLE = {'acdc': 3.9, 'abdomenct1k': 5.0, 'amos-ct': 6.4}
TRAINING_OVERHEAD_GIB = 1.0


def load_bundle(name):
    folder = ROOT / 'experiments' / name
    protocol = json.loads((folder / 'protocol.json').read_text())
    dataset = json.loads((folder / 'dataset.json').read_text())
    split = json.loads((folder / 'splits_final.json').read_text())[0]
    for group in ('train', 'val'):
        if not split[group] or len(split[group]) != len(set(split[group])):
            raise ValueError(f'Empty or duplicate {group} cases')
    if set(split['train']) & set(split['val']):
        raise ValueError('Training and validation cases overlap')
    return folder, protocol, dataset, split


def file_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    index = {}
    for path in folder.rglob('*.nii.gz'):
        index.setdefault(path.name, []).append(path.resolve())
    return index


def find_case(index, case, label=False):
    stems = [case]
    if case.startswith('ACDC_'):
        stems.append(case.removeprefix('ACDC_'))
    suffixes = ('_gt.nii.gz', '.nii.gz') if label else ('_0000.nii.gz', '.nii.gz')
    for stem in stems:
        for suffix in suffixes:
            matches = index.get(stem + suffix, [])
            if len(matches) > 1:
                raise ValueError(f'Ambiguous file: {stem + suffix}')
            if matches:
                return matches[0]
    raise FileNotFoundError(f'Missing {"label" if label else "image"} for {case}')


def mean_present(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(values) if values else None


def summarize(rows):
    labels = sorted({int(row['label']) for row in rows})
    per_label = {
        str(label): {
            metric: mean_present(row[metric] for row in rows if int(row['label']) == label)
            for metric in METRICS
        } for label in labels
    }
    return {
        'n_cases': len({row['case'] for row in rows}),
        'mean': per_label,
        'foreground_mean': {
            metric: mean_present(item[metric] for item in per_label.values()) for metric in METRICS
        },
    }


def write_results(output, rows):
    rows = sorted(rows, key=lambda row: (row['case'], int(row['label'])))
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(rows)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def surface_distances(reference, prediction, spacing):
    """Project surface metrics; equivalent to the experiment's surface-statistics script."""
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt

    union = reference | prediction
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(union.any(axis=tuple(i for i in range(3) if i != axis)))
        bounds.append(slice(max(int(occupied[0]) - 1, 0),
                            min(int(occupied[-1]) + 2, union.shape[axis])))
    ref, pred = reference[tuple(bounds)], prediction[tuple(bounds)]
    ref_surface, pred_surface = ref ^ binary_erosion(ref), pred ^ binary_erosion(pred)
    p2r = distance_transform_edt(~ref_surface, sampling=spacing)[pred_surface]
    r2p = distance_transform_edt(~pred_surface, sampling=spacing)[ref_surface]
    return (float(np.percentile(np.concatenate((p2r, r2p)), 95)),
            float(0.5 * (p2r.mean(dtype=np.float64) + r2p.mean(dtype=np.float64))))


def evaluate_case(task):
    import numpy as np
    import SimpleITK as sitk

    case, reference_path, prediction_path, labels = task
    ref_image, pred_image = sitk.ReadImage(str(reference_path)), sitk.ReadImage(str(prediction_path))
    if ref_image.GetSize() != pred_image.GetSize():
        raise ValueError(f'{case}: prediction and reference sizes differ')
    for attribute in ('GetSpacing', 'GetOrigin', 'GetDirection'):
        if not np.allclose(getattr(ref_image, attribute)(), getattr(pred_image, attribute)(),
                           atol=1e-5, rtol=0):
            raise ValueError(f'{case}: prediction and reference geometry differs ({attribute})')
    ref, pred = sitk.GetArrayFromImage(ref_image), sitk.GetArrayFromImage(pred_image)
    allowed = {0, *labels}
    if not set(np.unique(ref)).issubset(allowed) or not set(np.unique(pred)).issubset(allowed):
        raise ValueError(f'{case}: unexpected label values; check the published label mapping')
    spacing = tuple(reversed(ref_image.GetSpacing()))
    penalty = math.sqrt(sum(((size - 1) * step) ** 2
                            for size, step in zip(ref_image.GetSize(), ref_image.GetSpacing())))
    rows = []
    for label, name in labels.items():
        ref_mask, pred_mask = ref == label, pred == label
        n_ref, n_pred = int(ref_mask.sum()), int(pred_mask.sum())
        tp = int(np.count_nonzero(ref_mask & pred_mask))
        if not n_ref and not n_pred:
            dice = iou = hd95 = asd = None
            status = 'both_empty'
        else:
            dice, iou = 2 * tp / (n_ref + n_pred), tp / (n_ref + n_pred - tp)
            if not n_ref or not n_pred:
                hd95 = asd = penalty
                status = 'one_sided_empty'
            else:
                hd95, asd = surface_distances(ref_mask, pred_mask, spacing)
                status = 'both_present'
        rows.append(dict(case=case, label=label, label_name=name, Dice=dice, IoU=iou,
                         HD95_mm=hd95, ASD_mm=asd, status=status))
    return rows


def copy_matching(source, destination):
    """Install experiment metadata without replacing a different local experiment."""
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Different experiment metadata exists: {destination}. Use fresh paths.')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def prepare_training(folder, protocol, dataset, split, images, labels):
    for variable in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        if not os.environ.get(variable):
            raise ValueError(f'Set {variable} to a dedicated training directory')
    dataset_name = protocol['dataset']
    raw = Path(os.environ['nnUNet_raw']) / dataset_name
    preprocessed = Path(os.environ['nnUNet_preprocessed']) / dataset_name
    training_output = Path(os.environ['nnUNet_results']) / dataset_name / protocol['model_folder']
    if training_output.exists() and any(training_output.iterdir()):
        raise FileExistsError(f'Training output already exists: {training_output}. Use a fresh nnUNet_results directory.')
    cases = split['train'] + split['val']
    inputs = file_index(images)
    references = file_index(labels)
    files = [(case, find_case(inputs, case), find_case(references, case, label=True)) for case in cases]
    # Validate all metadata conflicts before adding any links. nnUNetv2_preprocess
    # reads dataset.json from the preprocessed folder, where planning would put it.
    copies = [(folder / 'dataset.json', raw / 'dataset.json'),
              (folder / 'dataset.json', preprocessed / 'dataset.json'),
              (folder / 'plans.json', preprocessed / 'nnUNetPlans.json'),
              (folder / 'splits_final.json', preprocessed / 'splits_final.json')]
    for source, destination in copies:
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Different experiment metadata exists: {destination}. Use fresh paths.')
    links = []
    for case, image, reference in files:
        links.extend(((image, raw / 'imagesTr' / f'{case}_0000.nii.gz'),
                      (reference, raw / 'labelsTr' / f'{case}.nii.gz')))
    expected = {destination for _, destination in links}
    for subfolder in ('imagesTr', 'labelsTr'):
        extra = set((raw / subfolder).glob('*.nii.gz')) - expected
        if extra:
            raise ValueError(f'Unexpected files in {raw / subfolder}; use a fresh dataset directory')
    for source, destination in links:
        if (destination.exists() or destination.is_symlink()) and destination.resolve() != source:
            raise FileExistsError(f'Different input exists: {destination}')
    for source, destination in copies:
        copy_matching(source, destination)
    for source, destination in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_symlink() and not destination.exists():
            destination.symlink_to(source)


def training_memory_error(alias, global_batch, gpus, gpu_gib):
    """Return advice if the per-GPU batch will not fit, otherwise None."""
    def needed(count):
        return global_batch // count * TRAINING_GIB_PER_SAMPLE[alias] + TRAINING_OVERHEAD_GIB

    if needed(gpus) <= gpu_gib:
        return None
    options = [count for count in range(gpus + 1, global_batch + 1)
               if not global_batch % count and needed(count) <= gpu_gib]
    advice = f'Use --gpus {options[0]} or more.' if options else 'Use GPUs with more memory.'
    return (f'Per-GPU batch {global_batch // gpus} needs about {needed(gpus):.0f} GiB; '
            f'the smallest selected GPU has {gpu_gib:.1f} GiB. {advice}')


def verify_cached_checkpoint(protocol):
    results = os.environ.get('nnUNet_results')
    if not results:
        raise ValueError('Set nnUNet_results to a writable checkpoint directory')
    model = Path(results) / protocol['dataset'] / protocol['model_folder']
    if not model.exists():
        return
    checkpoint = model / 'fold_0/checkpoint_best.pth'
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != protocol['checkpoint_sha256']:
        raise ValueError(f'{model} does not contain the released checkpoint. Use a fresh nnUNet_results directory.')
    bundle = ROOT / 'experiments' / protocol['alias']
    for filename in ('plans.json', 'dataset.json'):
        if not (model / filename).is_file() or json.loads((model / filename).read_text()) != json.loads((bundle / filename).read_text()):
            raise ValueError(f'{model / filename} differs from the released experiment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('--images', type=Path, help='Images directory (searched recursively)')
    parser.add_argument('--labels', required=True, type=Path, help='Reference labels directory')
    parser.add_argument('--output', type=Path, help='New directory for predictions and metrics')
    parser.add_argument('--predictions', type=Path, help='Evaluate existing predictions without a GPU')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--train', action='store_true', help='Prepare the published data/plans/split, preprocess, and train')
    parser.add_argument('--gpus', type=int, default=4, help='Training GPUs (reference: 4)')
    args = parser.parse_args()
    if args.workers < 1 or args.gpus < 1:
        parser.error('--workers and --gpus must be positive')
    if not args.predictions and not args.images:
        parser.error('--images is required unless --predictions is supplied')
    if args.train and args.predictions:
        parser.error('--train cannot be combined with --predictions')
    folder, protocol, dataset, split = load_bundle(args.dataset)
    if args.train:
        if protocol['training']['global_batch'] % args.gpus:
            parser.error('The reference global batch must divide evenly across GPUs')
        import torch

        if torch.cuda.device_count() < args.gpus:
            parser.error(f'--gpus {args.gpus} needs {args.gpus} visible CUDA GPUs; found {torch.cuda.device_count()}')
        gpu_gib = min(torch.cuda.get_device_properties(i).total_memory for i in range(args.gpus)) / 2**30
        message = training_memory_error(args.dataset, protocol['training']['global_batch'], args.gpus, gpu_gib)
        if message:
            parser.error(message)
        prepare_training(folder, protocol, dataset, split, args.images, args.labels)
        environment = dict(os.environ, nnUNet_compile='false',
                           SLIMFORMER_GLOBAL_BATCH=str(protocol['training']['global_batch']))
        # The trainer copies dataset_fingerprint.json into its output; preprocessing
        # uses the published plans, so extraction does not change the data.
        subprocess.run(['nnUNetv2_extract_fingerprint', '-d', str(protocol['dataset_id'])],
                       check=True, env=environment)
        subprocess.run(['nnUNetv2_preprocess', '-d', str(protocol['dataset_id']), '-c', '3d_fullres'],
                       check=True, env=environment)
        subprocess.run(['nnUNetv2_train', str(protocol['dataset_id']), '3d_fullres', '0',
                        '-tr', 'nnUNetTrainer_SlimFormerPlusPlus', '-num_gpus', str(args.gpus)],
                       check=True, env=environment)
        return
    if not args.output:
        parser.error('--output is required for evaluation')
    reference_index = file_index(args.labels)
    references = {case: find_case(reference_index, case, label=True) for case in split['val']}
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be a new or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.predictions:
        prediction_index = file_index(args.predictions)
        predictions = {case: find_case(prediction_index, case, label=True) for case in split['val']}
    else:
        verify_cached_checkpoint(protocol)
        input_index = file_index(args.images)
        inputs = {case: find_case(input_index, case) for case in split['val']}
        predictions_dir = args.output / 'predictions'
        with tempfile.TemporaryDirectory(prefix='sfpp-input-') as temporary:
            for case, source in inputs.items():
                (Path(temporary) / f'{case}_0000.nii.gz').symlink_to(source)
            subprocess.run(['bash', str(ROOT / 'predict.sh'), args.dataset, temporary, str(predictions_dir)],
                           check=True, env=dict(os.environ, nnUNet_compile='false'))
        verify_cached_checkpoint(protocol)
        predictions = {case: predictions_dir / f'{case}.nii.gz' for case in split['val']}
    label_names = {int(value): name for name, value in dataset['labels'].items() if value != 0}
    tasks = [(case, references[case], predictions[case], label_names) for case in split['val']]
    print(f'Evaluating {len(tasks)} fold-0 validation cases; workers={args.workers}', flush=True)
    rows = []
    if args.workers == 1:
        for task in tasks:
            rows.extend(evaluate_case(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(evaluate_case, tasks, chunksize=1):
                rows.extend(result)
    result = write_results(args.output, rows)
    reference = json.loads((folder / 'metrics.json').read_text())['foreground_mean']
    comparison = {metric: {'measured': result['foreground_mean'][metric], 'reference': reference[metric],
                           'difference': result['foreground_mean'][metric] - reference[metric]}
                  for metric in METRICS}
    (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print('Metric       Measured      Reference')
    for metric, values in comparison.items():
        print(f'{metric:10s} {values["measured"]:12.6f} {values["reference"]:12.6f}')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
