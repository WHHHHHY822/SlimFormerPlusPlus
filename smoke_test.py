"""Check trainer discovery and model construction; optionally run CUDA backward."""
import argparse
import importlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='Also check CUDA forward/backward')
    args = parser.parse_args()
    import torch
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    name = 'nnUNetTrainer_SlimFormerPlusPlus'
    module = importlib.import_module('nnunetv2.training.nnUNetTrainer.' + name)
    trainer = recursive_find_python_class(
        str(Path(nnunetv2.__path__[0]) / 'training' / 'nnUNetTrainer'),
        name, 'nnunetv2.training.nnUNetTrainer')
    assert trainer is getattr(module, name), 'nnU-Net trainer discovery failed'
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError('--cuda requires an available CUDA GPU')
    for stride in ((2, 2, 2), (1, 2, 2)):
        model = trainer.build_network_architecture(
            '', {'strides': [(1, 1, 1), stride], 'kernel_sizes': [(3, 3, 3)]},
            [], 1, 4, enable_deep_supervision=False)
        model.load_state_dict(model.state_dict(), strict=True)
        print(f'PASS construction: stride={stride}, parameters={sum(p.numel() for p in model.parameters())}')
        if args.cuda:
            model = model.cuda().train()
            x = torch.randn(1, 1, 32, 32, 32, device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                y = model(x)
                loss = y.float().square().mean()
            assert y.shape == (1, 4, 32, 32, 32), y.shape
            assert torch.isfinite(loss), 'Non-finite loss'
            loss.backward()
            bad = [n for n, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            assert not bad, f'Missing or non-finite gradients: {bad}'
            print(f'PASS CUDA forward/backward: stride={stride}')
            del x, y, loss
        del model
    print('PASS trainer discovery. This smoke test does not validate full training or dataset accuracy.')


if __name__ == '__main__':
    main()
