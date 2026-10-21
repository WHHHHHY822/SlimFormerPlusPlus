# Documentation

| Task | Guide |
| --- | --- |
| Install and troubleshoot | [Installation](INSTALLATION.md) |
| Obtain and prepare images | [Datasets](DATASETS.md) |
| Predict or train on your data | [Usage](USAGE.md) |
| Reproduce published results or benchmark | [Reproduction](../experiments/README.md) |
| Check tested environments and limitations | [Validation](VALIDATION.md) |
| Contribute code | [Contributing](../CONTRIBUTING.md) |

For maintainers: [release workflow](MAINTENANCE.md).

## Repository layout

| Location | Contents |
| --- | --- |
| Repository root | Model/trainer files and `predict.sh`, `reproduce.py`, `benchmark.py` |
| `tools/` | Installer, environment checks, dependency files and `tests/` |
| `experiments/` | Three dataset bundles and the historical `runtime.json` |
| `docs/` | User guides and maintenance notes |
| `assets/` | README figures |
| `licenses/` | Third-party license text |
