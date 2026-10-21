"""Measure SlimFormer++ parameters, GFLOPs, patch latency and peak GPU memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

from reproduce import DATASETS, ROOT


def measure_gflops(model, x):
    """Use the experiment's dispatcher counts plus analytic Mamba kernel work."""
    import torch
    from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn
    from torch.utils.flop_counter import FlopCounterMode

    analytic = {'mamba_conv1d': 0, 'mamba_scan': 0}

    def count_mamba(module, inputs, kwargs):
        tokens = inputs[0] if inputs else kwargs['hidden_states']
        batch, length, _ = tokens.shape
        if module.use_fast_path and causal_conv1d_fn is not None:
            analytic['mamba_conv1d'] += 2 * batch * length * module.d_inner * module.d_conv
        analytic['mamba_scan'] += 9 * batch * length * module.d_inner * module.d_state

    handles = [module.register_forward_pre_hook(count_mamba, with_kwargs=True)
               for module in model.modules() if isinstance(module, Mamba)]
    counter = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), counter:
            output = model(x)
        torch.cuda.synchronize()
        if not torch.isfinite(output).all():
            raise RuntimeError('Non-finite model output')
    finally:
        for handle in handles:
            handle.remove()
    counted = counter.get_total_flops()
    return {
        'gflops_total': (counted + sum(analytic.values())) / 1e9,
        'gflops_counted': counted / 1e9,
        'gflops_analytic': {name: count / 1e9 for name, count in analytic.items()},
        'flops_by_op': {str(op): int(count)
                        for op, count in counter.get_flop_counts().get('Global', {}).items()},
    }


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 1:
        parser.error('--warmup and --runs must be positive')
    if args.output.exists():
        parser.error('--output already exists; choose a new file')
    import torch
    from nnUNetTrainer_SlimFormerPlusPlus import SlimFormerPlusPlus

    if not torch.cuda.is_available():
        parser.error('An available CUDA GPU is required')
    bundle = ROOT / 'experiments' / args.dataset
    plans = json.loads((bundle / 'plans.json').read_text())['configurations']['3d_fullres']
    dataset = json.loads((bundle / 'dataset.json').read_text())
    architecture = plans['architecture']['arch_kwargs']
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    model = SlimFormerPlusPlus(
        in_channels=len(dataset['channel_names']), out_channels=len(dataset['labels']),
        first_stride=architecture['strides'][1], level0_kernel=architecture['kernel_sizes'][0],
    ).cuda().eval()
    x = torch.randn(1, len(dataset['channel_names']), *plans['patch_size'], device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(x)
            torch.cuda.synchronize()
            latencies.append(1000 * (time.perf_counter() - start))
            del output
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    versions = {}
    for name in ('torch', 'monai', 'einops', 'mamba-ssm', 'causal-conv1d'):
        versions[name] = importlib.metadata.version(name)
    record = {
        'dataset': plans.get('dataset_name', dataset['name']),
        'alias': args.dataset,
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(),
        'compute_capability': list(torch.cuda.get_device_capability()),
        'python': platform.python_version(), 'platform': platform.machine(),
        'versions': versions, 'cuda': torch.version.cuda,
        'trainer_sha256': hashlib.sha256((ROOT / 'nnUNetTrainer_SlimFormerPlusPlus.py').read_bytes()).hexdigest(),
        'plans_sha256': hashlib.sha256((bundle / 'plans.json').read_bytes()).hexdigest(),
        'benchmark_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mode': 'eval, eager, synthetic input, randomly initialized weights',
        'scope': 'single patch forward only; no preprocessing, sliding window, TTA or export',
        'batch': 1, 'patch': plans['patch_size'], 'autocast': 'bfloat16',
        'seed': 0, 'cudnn_benchmark': True,
        'warmup_iterations': args.warmup, 'measured_iterations': args.runs,
        'params': sum(parameter.numel() for parameter in model.parameters()),
        'latency_ms': latencies,
        'latency_ms_median': statistics.median(latencies),
        'latency_ms_p95': percentile(latencies, 0.95),
        'baseline_allocated_bytes': baseline,
        'peak_allocated_bytes': peak, 'peak_allocated_gib': peak / 2**30,
        **measure_gflops(model, x),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'{args.dataset}: {record["params"] / 1e6:.4f} M parameters, '
          f'{record["gflops_total"]:.2f} GFLOPs, '
          f'{record["latency_ms_median"]:.2f} ms median, '
          f'{record["peak_allocated_gib"]:.3f} GiB peak')
    print(args.output)


if __name__ == '__main__':
    main()
