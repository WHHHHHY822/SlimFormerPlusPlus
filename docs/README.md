[← Project home](../README.md)

# Documentation

Install SlimFormer++, prepare your data, then predict or reproduce a released
checkpoint. Choose a guide below for the step you need.

**Start here:** [Install](INSTALLATION.md) → [Prepare data](DATASETS.md) →
[Predict](USAGE.md#predict-with-a-released-checkpoint)

## Guides

| Task | Guide |
| --- | --- |
| Install and troubleshoot | [Installation](INSTALLATION.md) |
| Obtain and prepare images | [Datasets](DATASETS.md) |
| Predict or train on your data | [Usage](USAGE.md) |
| Evaluate released checkpoints or measure efficiency | [Reproduction](../experiments/README.md) |
| Check tested environments and limitations | [Validation](VALIDATION.md) |
| Contribute code | [Contributing](../CONTRIBUTING.md) |
| Maintain checkpoint releases | [Release workflow](MAINTENANCE.md) |

## Repository layout

| Location | Contents |
| --- | --- |
| Repository root | Model/trainer files and `predict.sh`, `reproduce.py`, `benchmark.py` |
| [tools/](../tools/) | Installer, environment checks, dependency files and tests |
| [experiments/](../experiments/) | Three dataset bundles and the historical `runtime.json` |
| [docs/](./) | User guides and maintenance notes |
| [assets/](../assets/) | Paper comparison figures |
| [licenses/](../licenses/) | Third-party license text |
