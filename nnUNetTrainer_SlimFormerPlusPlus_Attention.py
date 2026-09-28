"""SlimFormer++ attention variant: window self-attention in encoder 1.

Standalone model and nnU-Net v2 trainer, independent of
``nnUNetTrainer_SlimFormerPlusPlus.py``. The two ``SelectiveMixer`` token mixers
of encoder 1 are replaced by ``WindowSelfAttention``; every other module, the
training schedule and the optimizer are copied unchanged from SlimFormer++.
Dependencies: torch, einops, monai, nnunet_mednext; nnunetv2 for training.
No mamba_ssm or causal_conv1d is needed.
"""
from __future__ import annotations
import math
import os
import time
import signal
from typing import Sequence
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

MEDNEXT_EXP_R = 2
MEDNEXT_KERNEL = 3

def window_partition(x, window_size):
   B, D, H, W, C = x.shape
   wd, wh, ww = window_size
   x = x.view(
       B,
       D // wd, wd,
       H // wh, wh,
       W // ww, ww,
       C
   )
   x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
   windows = x.view(-1, wd * wh * ww, C)
   return windows


def window_reverse(windows, window_size, B, D, H, W, C):
   wd, wh, ww = window_size
   x = windows.view(
       B,
       D // wd,
       H // wh,
       W // ww,
       wd, wh, ww,
       C
   )
   x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
   x = x.view(B, D, H, W, C)
   return x


class DropPath(nn.Module):
   def __init__(self, drop_prob=0.):
       super().__init__()
       self.drop_prob = drop_prob

   def forward(self, x):
       if self.drop_prob == 0. or not self.training:
           return x
       keep_prob = 1 - self.drop_prob
       shape = (x.shape[0],) + (1,) * (x.ndim - 1)
       random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
       random_tensor.floor_()
       return x.div(keep_prob) * random_tensor


class Scale(nn.Module):
   def __init__(self, dim, init_value=1e-6):
       super().__init__()
       self.scale = nn.Parameter(init_value * torch.ones(1, 1, dim))

   def forward(self, x):
       return x * self.scale


class Mlp(nn.Module):
   def __init__(self, dim, hidden_dim=None, drop=0.):
       super().__init__()
       hidden_dim = hidden_dim or dim * 4
       self.fc1 = nn.Linear(dim, hidden_dim)
       self.act = nn.GELU()
       self.fc2 = nn.Linear(hidden_dim, dim)
       self.drop = nn.Dropout(drop)

   def forward(self, x):
       x = self.fc1(x)
       x = self.act(x)
       x = self.drop(x)
       x = self.fc2(x)
       x = self.drop(x)
       return x


class WindowSelfAttention(nn.Module):
    """Multi-head self-attention among the tokens of each window.

    Takes the place of SlimFormer++'s ``SelectiveMixer``: the input is
    ``(B*num_windows, L, C)`` from 8x8x8 window partitioning, every token
    attends to all tokens of its window, and the output has the same shape.
    There is no positional bias, and dropout is applied to the output only, as
    in ``SelectiveMixer``.

    The head dimension must be a multiple of 8 so that PyTorch's fused
    attention kernels apply; otherwise ``scaled_dot_product_attention`` can
    fall back to its math implementation, which stores the full
    windows x heads x L x L score tensor. At width 48 this gives 3 heads of 16.
    """

    def __init__(self, dim, head_dim=16, qkv_bias=False, drop=0.):
        super().__init__()
        if dim % head_dim or head_dim % 8:
            raise ValueError(
                f"head_dim must divide dim and be a multiple of 8, got dim={dim}, head_dim={head_dim}"
            )
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"WindowSelfAttention expects (B*num_windows, L, C), got {tuple(x.shape)}. "
                "The stage must run with window partitioning."
            )
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.dropout(self.proj(x))


class AlignedMedNeXtUpBlock(MedNeXtUpBlock):
    """``MedNeXtUpBlock`` that lands on an exact target size, never resampling.

    The stock block always emits ``2*in``.  Here the caller passes the size it
    needs (the skip's), and the one-voxel front pad is applied per axis only when
    that size asks for it.  A ceil-downsampled encoder can only ever ask for
    ``2*in - 1`` or ``2*in``, so this is exact; the interpolate fallback exists
    purely so an unexpected patch size degrades instead of crashing.
    """

    def forward(self, x, target_size):
        x1 = MedNeXtBlock.forward(self, x)  # transposed conv path -> 2*in - 1
        x1 = self._fit(x1, target_size)
        if self.resample_do_res:
            x1 = x1 + self._fit(self.res_conv(x), target_size)
        return x1

    @staticmethod
    def _fit(t, target_size):
        target_size = tuple(int(s) for s in target_size)
        deltas = [tgt - cur for tgt, cur in zip(target_size, t.shape[2:])]
        if any(d > 1 for d in deltas):
            return F.interpolate(t, size=target_size, mode="trilinear", align_corners=False)
        if any(d > 0 for d in deltas):
            pad = []
            for d in reversed(deltas):  # F.pad takes the axes back to front
                pad += [max(d, 0), 0]
            t = F.pad(t, pad)
        return t[:, :, :target_size[0], :target_size[1], :target_size[2]]


class AlignedMedNeXtUpStage(nn.Module):
    """MedNeXt Up block sized to the skip + additive fusion + one MedNeXt block.

    Identical to ``MedNeXtUpStage`` except that the skip is used as-is instead of
    being trilinearly resampled onto a blindly doubled grid.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = AlignedMedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=MEDNEXT_EXP_R,
            kernel_size=MEDNEXT_KERNEL,
            do_res=True,
        )
        self.dec_block = MedNeXtBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=MEDNEXT_EXP_R,
            kernel_size=MEDNEXT_KERNEL,
            do_res=True,
        )

    def forward(self, x, skip):
        x = self.up(x, skip.shape[2:])
        return self.dec_block(x + skip)


def _triple(values: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(v) for v in values)
    if len(result) != 3 or any(v < 1 for v in result):
        raise ValueError(f"G1 requires a positive 3D {name}, got {values!r}")
    return result


class PlansAwareMedNeXtDownBlock(MedNeXtDownBlock):
    """MedNeXt down block with a plans-provided anisotropic stride."""

    def __init__(self, in_channels: int, out_channels: int, stride: Sequence[int]):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=MEDNEXT_EXP_R,
            kernel_size=MEDNEXT_KERNEL,
            do_res=True,
        )
        stride = _triple(stride, "downsampling stride")
        self.conv1 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=MEDNEXT_KERNEL,
            stride=stride,
            padding=MEDNEXT_KERNEL // 2,
            groups=in_channels,
        )
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)


def pixel_shuffle3d(x, out_channels, scale):
    rd, rh, rw = scale
    b, _, d, h, w = x.shape
    return (x.reshape(b, out_channels, rd, rh, rw, d, h, w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .reshape(b, out_channels, d * rd, h * rh, w * rw))


class PlansSubPixelUp(nn.Module):
    """DW k3 + channelwise GN + ICNR pointwise projection on the coarse grid.

    Per-axis scale follows the inverse encoder stride. Ceil-downsampling can
    leave at most scale-1 excess voxels; crop only the tail. Unexpected shapes
    fail explicitly rather than silently interpolating and hiding a bad plan.
    """
    def __init__(self, in_channels, out_channels, scale=(2, 2, 2)):
        super().__init__()
        self.scale = tuple(int(s) for s in scale)
        if len(self.scale) != 3 or any(s not in (1, 2) for s in self.scale):
            raise ValueError(f"Unsupported scale {self.scale}")
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, padding=1,
                               groups=in_channels)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.conv3 = nn.Conv3d(in_channels, out_channels * math.prod(self.scale), 1)
        with torch.no_grad():
            repeats = math.prod(self.scale)
            self.conv3.weight.copy_(self.conv3.weight[:out_channels].repeat_interleave(repeats, 0))
            self.conv3.bias.copy_(self.conv3.bias[:out_channels].repeat_interleave(repeats, 0))

    def forward(self, x, target_size):
        x = pixel_shuffle3d(self.conv3(self.norm(self.conv1(x))), self.out_channels, self.scale)
        for actual, target, scale in zip(x.shape[2:], target_size, self.scale):
            if not 0 <= actual - target < scale:
                raise ValueError(f"Invalid ceil-downsample inverse: {x.shape[2:]} -> {target_size}")
        return x[:, :, :target_size[0], :target_size[1], :target_size[2]]


class AugmentedMixer(nn.Module):
    """R5 CapFree；输入和输出均为 (B, D*H*W, C)。"""

    NO_DECAY = ("theta", "relation_bias")
    TAPS = (
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    )
    FRACTIONS = (0.25, 0.5, 0.75, 1.0)

    def __init__(
        self, dim=384, descriptor_dim=16, groups=8,
        temperature=0.2, eps=1e-6,
    ):
        super().__init__()
        if dim % groups:
            raise ValueError("dim 必须能被 groups 整除")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.dim = dim
        self.groups = groups
        self.temperature = temperature
        self.eps = eps

        self.depthwise = nn.Conv3d(
            dim, dim, 3, padding=1, groups=dim, bias=False
        )
        self.descriptor = nn.Linear(dim, descriptor_dim, bias=False)
        self.theta = nn.Parameter(torch.full((groups,), 0.1))
        self.relation_bias = nn.Parameter(torch.zeros(6, 4))

    @staticmethod
    def _shift(x, offset):
        """out[..., i] = x[..., i + offset]，越界补零。"""
        out = x
        for axis, step in enumerate(offset):
            if step == 0:
                continue
            dim = x.ndim - 3 + axis
            length = out.shape[dim]
            if abs(step) >= length:
                return torch.zeros_like(x)
            pad = [0] * 6
            slot = 2 * (2 - axis)
            if step > 0:
                out = out.narrow(dim, step, length - step)
                pad[slot + 1] = step
            else:
                out = out.narrow(dim, 0, length + step)
                pad[slot] = -step
            out = F.pad(out, pad)
        return out

    @classmethod
    def _candidates(cls, tap, shape):
        axis = next(i for i, step in enumerate(tap) if step)
        length = shape[axis]
        seen = set()
        candidates = []
        for slot, fraction in enumerate(cls.FRACTIONS):
            radius = max(2, int(math.floor(fraction * (length - 1) + 0.5)))
            if radius in seen or radius > length - 1:
                continue
            seen.add(radius)
            offset = [0, 0, 0]
            offset[axis] = tap[axis] * radius
            candidates.append((slot, tuple(offset)))
        return candidates

    def forward(self, tokens, spatial_shape):
        d, h, w = spatial_shape
        b, n, c = tokens.shape
        if n != d * h * w or c != self.dim:
            raise ValueError("tokens 形状与 spatial_shape 或 dim 不匹配")

        x = rearrange(
            tokens, "b (d h w) c -> b c d h w", d=d, h=h, w=w
        )
        local = self.depthwise(x)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            s = F.normalize(
                self.descriptor(tokens.float()), dim=-1, eps=self.eps
            )
        s = rearrange(s, "b (d h w) k -> b k d h w", d=d, h=h, w=w)

        values = x.to(local.dtype)
        beta = self.theta.to(local.dtype)[None, :, None, None, None, None]
        kernel = self.depthwise.weight.reshape(c, 27)
        correction = torch.zeros_like(
            local.reshape(b, self.groups, c // self.groups, d, h, w)
        )
        ones = torch.ones((1, 1, d, h, w), device=x.device)
        used = False

        for tap_slot, tap in enumerate(self.TAPS):
            candidates = self._candidates(tap, spatial_shape)
            if not candidates:
                continue
            used = True
            masks, samples, logits = [], [], []
            for fraction_slot, offset in candidates:
                masks.append(self._shift(ones, offset) > 0)
                samples.append(self._shift(values, offset))
                cosine = (s * self._shift(s, offset)).sum(dim=1, keepdim=True)
                logits.append(
                    cosine / self.temperature
                    + self.relation_bias[tap_slot, fraction_slot]
                )

            valid = torch.cat(masks, dim=1)
            any_valid = valid.any(dim=1, keepdim=True)
            scores = torch.cat(logits, dim=1).float()
            blocked = torch.full_like(scores, torch.finfo(scores.dtype).min / 4)
            weights = torch.where(valid, scores, blocked).softmax(dim=1)
            weights = weights.to(values.dtype)

            remote = torch.zeros_like(values)
            for i, sample in enumerate(samples):
                remote = remote + sample * weights[:, i:i + 1]

            delta = (remote - self._shift(values, tap)) * any_valid
            index = (tap[0] + 1) * 9 + (tap[1] + 1) * 3 + tap[2] + 1
            weight = kernel[:, index].reshape(1, c, 1, 1, 1).to(local.dtype)
            correction = correction + (weight * delta).reshape(
                b, self.groups, c // self.groups, d, h, w
            ) * beta

        y = local + correction.reshape_as(local)
        if not used:
            # 退化网格仍让所有参数进入计算图，兼容 DDP。
            keepalive = self.theta.sum()
            for name, parameter in self.named_parameters():
                if name != "theta":
                    keepalive = keepalive + parameter.sum()
            y = y + 0.0 * keepalive.to(y.dtype)

        return rearrange(y, "b c d h w -> b (d h w) c")

class MixerBlock(nn.Module):
    """MLP residual always present; the mixer residual exists only if requested."""
    def __init__(self, dim, mixer=None, drop=0.2, drop_path=0.2):
        super().__init__()
        self.prune_ln1_token_mixer = mixer is None
        self.prune_ln2_mlp = False
        self.norm1 = nn.LayerNorm(dim) if mixer is not None else nn.Identity()
        self.token_mixer = mixer if mixer is not None else nn.Identity()
        self.res_scale1 = Scale(dim, 1.0) if mixer is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if mixer is not None else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, drop=drop)
        self.res_scale2 = Scale(dim, 1.0)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x, spatial_shape=None):
        if not self.prune_ln1_token_mixer:
            z = self.norm1(x)
            z = self.token_mixer(z) if spatial_shape is None else self.token_mixer(z, spatial_shape)
            x = x + self.drop_path1(self.res_scale1(z))
        return x + self.drop_path2(self.res_scale2(self.mlp(self.norm2(x))))


class EncoderStage(nn.Module):
    """Locally refined skip also feeds the deeper trunk, as in the L1 parent."""
    def __init__(self, channels, attention=False, drop=0.2, drop_path=0.2):
        super().__init__()
        self.window_size = (8, 8, 8)
        self.use_windows = attention
        self.metaformer = nn.Sequential(*[
            MixerBlock(channels, WindowSelfAttention(channels, drop=drop) if attention else None,
                       drop, drop_path) for _ in range(2)
        ])
        self.skip_conv = UnetrBasicBlock(3, channels, channels, 3, 1, "instance", res_block=True)
        self.down = MedNeXtDownBlock(channels, channels * 2, exp_r=2, kernel_size=3, do_res=True)

    def mix(self, x):
        if not self.use_windows:
            return self.metaformer(x)
        b, d, h, w, c = x.shape
        pad = tuple((8 - size % 8) % 8 for size in (d, h, w))
        if any(pad):
            x = F.pad(x.permute(0, 4, 1, 2, 3), (0, pad[2], 0, pad[1], 0, pad[0])).permute(0, 2, 3, 4, 1)
        _, dp, hp, wp, _ = x.shape
        x = window_reverse(self.metaformer(window_partition(x, self.window_size)),
                           self.window_size, b, dp, hp, wp, c)
        return x[:, :d, :h, :w].contiguous() if any(pad) else x

    def forward(self, x):
        x = self.mix(rearrange(x, "b c d h w -> b d h w c"))
        skip = self.skip_conv(rearrange(x, "b d h w c -> b c d h w"))
        return self.down(skip), skip


class AugmentedBottleneck(nn.Module):
    def __init__(self, channels=384, drop=0.2, drop_path=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock(channels, AugmentedMixer(channels), drop, drop_path)])

    def forward(self, x):
        d, h, w = x.shape[2:]
        x = rearrange(x, "b c d h w -> b (d h w) c")
        for block in self.blocks:
            x = block(x, (d, h, w))
        return rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)


class SlimFormerPlusPlusAttention(nn.Module):
    """SlimFormer++ with window self-attention, not selective mixing, in encoder1."""
    def __init__(self, in_channels=1, out_channels=4, first_stride=(2, 2, 2),
                 level0_kernel=(3, 3, 3), drop=0.2, drop_path=0.2):
        super().__init__()
        self.first_stride = _triple(first_stride, "first stride")
        self.level0_kernel = _triple(level0_kernel, "level0 kernel")
        self.base_channels = 48
        k = self.level0_kernel
        padding = tuple(v // 2 for v in k)
        self.encoder0 = nn.Sequential(
            nn.Conv3d(in_channels, 32, k, padding=padding),
            nn.InstanceNorm3d(32, eps=1e-5, affine=True), nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, k, padding=padding),
            nn.InstanceNorm3d(32, eps=1e-5, affine=True), nn.LeakyReLU(inplace=True),
        )
        self.down0 = PlansAwareMedNeXtDownBlock(32, 48, self.first_stride)
        self.pos_drop = nn.Dropout(drop)
        self.encoder1 = EncoderStage(48, True, drop, drop_path)
        self.encoder2 = EncoderStage(96, False, drop, drop_path)
        self.encoder3 = EncoderStage(192, False, drop, drop_path)
        self.bottleneck = AugmentedBottleneck(384, drop, drop_path)
        self.decoder3 = AlignedMedNeXtUpStage(384, 192)
        self.decoder2 = AlignedMedNeXtUpStage(192, 96)
        self.decoder1 = AlignedMedNeXtUpStage(96, 48)
        self.decoder1.up = PlansSubPixelUp(96, 48)
        self.decoder0 = AlignedMedNeXtUpStage(48, 32)
        self.decoder0.up = PlansSubPixelUp(48, 32, self.first_stride)
        self.decoder0.dec_block = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1),
            nn.InstanceNorm3d(32, eps=1e-5, affine=True), nn.LeakyReLU(inplace=True),
        )
        self.seg_head = UnetOutBlock(3, 32, out_channels)

    def forward(self, x):
        s0 = self.encoder0(x)
        x = self.pos_drop(self.down0(s0))
        x, s1 = self.encoder1(x)
        x, s2 = self.encoder2(x)
        x, s3 = self.encoder3(x)
        x = self.bottleneck(x)
        x = self.decoder3(x, s3)
        x = self.decoder2(x, s2)
        x = self.decoder1(x, s1)
        return self.seg_head(self.decoder0(x, s0))


# The model above is usable without nnU-Net. The trainer below requires nnunetv2.
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("nnunetv2"):
        raise
else:
    class nnUNetTrainer_SlimFormerPlusPlus_Attention(nnUNetTrainer):
        """SlimFormer++ training schedule; encoder 1 uses window self-attention."""
        INIT_SEED = 20260921

        def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device=device)
            self.enable_deep_supervision = False
            self.num_epochs = 1200
            self.num_iterations_per_epoch = 250
            self.num_val_iterations_per_epoch = 50
            # The site's nnU-Net has an optional sample-budget hook; do not
            # silently reduce optimizer steps when the four-GPU batch increases.
            self.train_samples_per_epoch = None

        def _set_batch_size_and_oversample(self):
            batch = os.environ.get("SLIMFORMER_GLOBAL_BATCH")
            if batch is not None:
                batch = int(batch)
                world = torch.distributed.get_world_size() if self.is_ddp else 1
                if batch < world or batch % world:
                    raise ValueError("SLIMFORMER_GLOBAL_BATCH must divide evenly across ranks")
                self.configuration_manager.configuration["batch_size"] = batch
            super()._set_batch_size_and_oversample()

        @staticmethod
        def build_network_architecture(architecture_class_name, arch_init_kwargs,
                arch_init_kwargs_req_import, num_input_channels, num_output_channels,
                enable_deep_supervision=False):
            if tuple(arch_init_kwargs["strides"][0]) != (1, 1, 1):
                raise ValueError("Expected full-resolution first stage")
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20260921)
                return SlimFormerPlusPlusAttention(
                    num_input_channels, num_output_channels,
                    first_stride=arch_init_kwargs["strides"][1],
                    level0_kernel=arch_init_kwargs["kernel_sizes"][0],
                )

        def set_deep_supervision_enabled(self, enabled):
            pass

        def on_epoch_end(self):
            # Optional walltime integration with the site's graceful nnU-Net
            # runner. Normal standalone training does not set this variable.
            deadline = float(os.environ.get("SLIMFORMER_STOP_AT", "inf"))
            if time.time() >= deadline and hasattr(self, "_graceful_stop_requested"):
                self._graceful_stop_requested = True
                self._graceful_stop_signal = signal.SIGUSR1
            super().on_epoch_end()

        def configure_optimizers(self):
            model = self.network
            while hasattr(model, "module"):
                model = model.module
            exempt = {id(p) for name, p in model.bottleneck.blocks[0].token_mixer.named_parameters()
                      if any(name.startswith(s) for s in AugmentedMixer.NO_DECAY)}
            groups = [
                {"params": [p for p in self.network.parameters() if id(p) not in exempt],
                 "weight_decay": self.weight_decay},
                {"params": [p for p in self.network.parameters() if id(p) in exempt],
                 "weight_decay": 0.0},
            ]
            optimizer = torch.optim.SGD(groups, self.initial_lr, momentum=0.99, nesterov=True)
            return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
