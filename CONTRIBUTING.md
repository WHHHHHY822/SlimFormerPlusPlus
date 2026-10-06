# Contributing

Please discuss architecture or training-behavior changes in an issue before
opening a pull request. Include the motivation and the affected configurations.

Use a dedicated environment and follow the installation instructions in the
README. Run the checks that apply to your change:

```bash
# No model dependencies needed:
python -m unittest discover -s tests -v
python -m compileall -q install.py check_environment.py smoke_test.py nnUNetTrainer_SlimFormerPlusPlus.py \
    nnUNetTrainer_SlimFormerPlusPlus_Attention.py tests

# With model dependencies and an installed trainer:
python check_environment.py
python install.py --check
python smoke_test.py
# Requires an available CUDA GPU:
python smoke_test.py --cuda
# For the attention variant, add --attention to the three commands above.
```

If model code changed, `install.py` will refuse to overwrite a different
installed trainer. Preserve the old version before explicitly removing it and
installing your updated copy in the dedicated test environment.

Keep checkpoint parameter names and tensor shapes stable unless an intentional
compatibility change is documented. State which tests ran, the runtime versions,
and any unavailable checks. GitHub CI covers syntax and installer behavior; it
does not run GPU training.

Do not include dataset images, patient identifiers, checkpoints, access tokens,
or machine-specific credentials in issues or pull requests. For runtime bugs,
include a minimal reproduction, a sanitized traceback, dependency versions,
GPU model, CUDA version, and the input shape/plan settings.

Submit only contributions you have the right to share under the project's
noncommercial research terms. Preserve third-party copyright and license
notices. No contributor agreement or new licensing policy is implied here.
