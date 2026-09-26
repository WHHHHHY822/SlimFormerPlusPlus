# Publishing a source release

The repository root contains the complete SlimFormer++ source release.
Weights and datasets are excluded. The project license permits noncommercial
research only; do not label the entire repository as Apache-2.0 or OSI-approved.

Before publishing updates:

1. Preserve project and upstream copyright and license notices.
2. Run the checks in CONTRIBUTING.md and update [VALIDATION.md](VALIDATION.md)
   with the checks actually completed.
3. Review the staged files and keep credentials, datasets, logs and weights out
   of Git history.

```bash
git status --short
git add .
git diff --cached --stat
git diff --cached --check
# Review the staged changes before committing and pushing.
git commit -m "Update SlimFormer++ research source release"
git push origin main
```

GitHub Actions checks Python syntax and installer behavior. It does not replace
CUDA or dataset-specific training validation. Publish trained weights separately
only after their export format, evaluation results and checksums are verified.
