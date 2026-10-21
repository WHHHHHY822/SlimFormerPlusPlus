# SlimFormer++ optimization roadmap

Historical planning notes, retained for reference. Reproduction bundles and patch
benchmarks have since been published; see the [current guide](../../experiments/README.md).

This review concerns the standalone SlimFormer++ repository. It is based on
the local source and recorded validation; no new GPU benchmark or accuracy
experiment was run for this review. The priorities below are proposed work,
not claims of measured speedups.

## 1. Make the published results reproducible

The README already provides pinned dependencies, checked checkpoint downloads
and training commands. The largest remaining gap is the experiment protocol:
`docs/VALIDATION.md` explicitly separates installation checks from accuracy,
efficiency and multi-GPU validation.

For each released dataset, publish a versioned experiment manifest containing:

- The exact `plans.json`, `dataset.json` and case IDs in `splits_final.json`,
  where redistribution is permitted; otherwise provide a split-generation
  script and hashes that allow users to verify their local files.
- The trainer source hash, checkpoint hash, software versions, fold, seed,
  optimizer settings, global batch size, GPU count and training duration.
- Evaluation settings: best versus final checkpoint, TTA, ensemble,
  postprocessing, label aggregation and the commands used to obtain metrics.
- Per-case metrics and a machine-readable aggregate, with the evaluation
  implementation and its version.

`runtime.json` preserves historical runtime and shape information; it is not
a complete reproduction manifest. In particular, the trainer runs 1,200
epochs with 250 optimizer steps per epoch. Its optional
`SLIMFORMER_GLOBAL_BATCH` changes how many samples those steps see. Record
that value alongside the number of ranks before comparing training budgets.

Acceptance: a user can reproduce one released dataset's evaluation from the
public checkpoint and a documented local dataset without private scripts.

## 2. Extend checks from installation to model behavior

The current source CI runs compilation and installer tests.
`smoke_test.py --cuda` covers construction and forward/backward on a
`32 x 32 x 32` patch for two first strides. It does not cover real dataset
patches, odd spatial dimensions, checkpoint round trips through the predictor
or multi-GPU training.

Add a separate optional model-test job in a supported dependency environment,
then a GPU job where hardware is available. Useful contracts include:

- `window_partition` / `window_reverse` round trips and padded encoder windows.
- `PlansSubPixelUp` inverse shapes for both `(2, 2, 2)` and `(1, 2, 2)`,
  including odd dimensions and explicitly rejected shapes.
- `AugmentedMixer._shift` boundaries, duplicate candidate radii, degenerate
  grids, finite outputs and gradients, and masking where no remote tap is valid.
- Strict checkpoint loading and an nnU-Net predictor round trip for both
  first strides. Use a released checkpoint for compatibility verification;
  loading a model's own state only checks its current internal consistency.
- A short two-rank training run that checks gradients and effective global
  batch size before documenting DDP support.

Keep lightweight installer checks independent of CUDA dependencies. Preserve
existing parameter names, shapes and checkpoint compatibility when adding tests.

## 3. Profile the actual inference and training workloads

Add a benchmark command that emits JSON instead of relying only on figures.
Use the dataset-specific patch sizes in `runtime.json`, plus both original
and attention models. Record hardware, runtime versions, dtype, batch size,
input shape, warmup count, measured iterations and compile settings.

Measure evaluation forward latency, training forward/backward latency and peak
allocated GPU memory separately. Synchronize CUDA around timings, exclude
warmup and compilation from steady-state latency, and report a distribution
such as median and p95. Also measure end-to-end sliding-window prediction;
patch forward latency does not include preprocessing, TTA, blending or export.

Source inspection suggests two places to profile first:

- `AugmentedMixer.forward` builds candidate masks, shifted feature tensors,
  descriptors and scores in nested loops over six directions and up to four
  radii. Profile allocations and kernel launches. Candidate geometry and
  validity masks depend on shape rather than weights; bounded shape/device
  caches or grouped operations are possible experiments. Cache no tensors
  attached to an autograd graph, and do not persist caches in checkpoints.
- `EncoderStage.mix` pads to `8 x 8 x 8` windows and changes tensor layout;
  `window_partition` and `window_reverse` materialize contiguous layouts.
  Measure this cost separately from selective scanning before changing layouts.

For every optimized implementation, compare outputs and gradients with the
current reference on anisotropic and boundary cases. Retain the descriptor
and score precision needed for numerical stability. Keep each optimization
only if measured latency or memory improves on representative shapes.

## 4. Evaluate deployment choices independently

The README disables `nnUNet_compile` for its reference training command.
Benchmark eager and compiled execution in separate runs before advertising
compile support. Record graph breaks, first-run cost, steady-state behavior
and any numerical differences for Mamba and the attention variant separately.

If memory limits training, evaluate activation checkpointing around encoder
blocks as an optional setting, measuring the memory reduction and added
compute. Treat alternate precision, layout, window size and token-mixer
changes as experiments requiring validation, with architecture changes
reported separately from implementation speedups.

## Suggested order

1. Publish the evaluation manifests and commands for existing checkpoints.
2. Add shape, boundary and predictor compatibility checks.
3. Capture a reproducible benchmark baseline.
4. Optimize measured hotspots and publish before/after results.

The public model source is hashed in `docs/VALIDATION.md`. Any future source
change should update its validation record and recheck released checkpoints.
