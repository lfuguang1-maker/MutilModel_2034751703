# 感谢多模态群友 Accepted 对本文件复现以及分享
# 论文方法：DACFusion
# 来源：paper_ports 复现包（本地适配实现）
"""Dual-stream DACFusion backbone adapted to the repository's concatenated-modal input.

The supplied DACFusion fork accepts ``forward(rgb, ir)`` and performs a costly 1024-square dummy
forward in ``__init__``.  This implementation keeps its overlap-patch hierarchy, DAC spatial gate,
late cross-modal blocks and attention fusion, while splitting the routed tensor internally and
declaring output channels without a constructor-time forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * (torch.rand(shape, device=x.device, dtype=x.dtype) < keep) / keep


class _DWMLP(nn.Module):
    def __init__(self, channels: int, ratio: float, drop: float):
        super().__init__()
        hidden = int(channels * ratio)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, channels, 1),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.layers(x)


class _Align(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class _DACGate(nn.Module):
    def __init__(self, channels: int, cross_modal: bool):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.cross_modal = cross_modal
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.context = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.local_reduce = nn.Conv2d(channels, hidden, 1)
        self.context_reduce = nn.Conv2d(channels, hidden, 1)
        self.local_align = _Align(hidden) if cross_modal else nn.Identity()
        self.context_align = _Align(hidden) if cross_modal else nn.Identity()
        self.selector = nn.Conv2d(2, 2, 7, padding=3)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x, other=None):
        local = self.local_reduce(self.local(x))
        context_source = other if self.cross_modal and other is not None else x
        context = self.context_reduce(self.context(context_source))
        local, context = self.local_align(local), self.context_align(context)
        joined = torch.cat((local, context), dim=1)
        stats = torch.cat((joined.mean(1, keepdim=True), joined.amax(1, keepdim=True)), dim=1)
        weights = self.selector(stats).sigmoid()
        attention = self.expand(local * weights[:, :1] + context * weights[:, 1:])
        return x * attention


class _DACAttention(nn.Module):
    def __init__(self, channels: int, cross_modal: bool):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels, 1)
        self.gate = _DACGate(channels, cross_modal)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, other=None):
        shortcut = x
        x = F.gelu(self.in_proj(x))
        return shortcut + self.out_proj(self.gate(x, other))


class _DACBlock(nn.Module):
    def __init__(self, channels: int, ratio: float, drop: float, drop_path: float, cross_modal: bool):
        super().__init__()
        self.cross_modal = cross_modal
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.attn = _DACAttention(channels, cross_modal)
        self.mlp = _DWMLP(channels, ratio, drop)
        self.drop_path = _DropPath(drop_path)
        self.scale1 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))
        self.scale2 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x, other=None):
        x = x + self.drop_path(self.scale1 * self.attn(self.norm1(x), self.norm1(other) if other is not None else None))
        return x + self.drop_path(self.scale2 * self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first: bool):
        super().__init__()
        kernel, stride = (7, 4) if first else (3, 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class _AttentionFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.rgb = nn.Conv2d(channels, channels, 3, padding=1)
        self.aux = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, rgb, aux):
        joined = torch.cat((self.rgb(rgb), self.aux(aux)), dim=1)
        return self.out(joined * self.attention(joined))


class DACFusionBackbone(nn.Module):
    """Four-level DACFusion backbone for an RGB + auxiliary concatenated input.

    Args:
        input_channels: Total routed input channels; injected by ``parse_model``.
        rgb_channels: Number of leading RGB channels.
        variant: ``tiny`` matches ``DACnet_t_mul``; ``small`` matches ``DACnet_s_mul``.
        aux_channels: Remaining channels by default, or an explicit count for validation.
    """

    _VARIANTS = {
        "tiny": ([32, 64, 160, 256], [3, 3, 5, 2], [8, 8, 4, 4]),
        "small": ([64, 128, 256, 512], [2, 2, 1, 1], [8, 8, 4, 4]),
    }

    def __init__(
        self,
        input_channels: int,
        rgb_channels: int = 3,
        variant: str = "tiny",
        aux_channels: int | None = None,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in self._VARIANTS:
            raise ValueError(f"DACFusionBackbone variant must be one of {tuple(self._VARIANTS)}, got {variant!r}")
        inferred_aux = int(input_channels) - int(rgb_channels)
        aux_channels = inferred_aux if aux_channels is None else int(aux_channels)
        if rgb_channels <= 0 or aux_channels <= 0 or rgb_channels + aux_channels != input_channels:
            raise ValueError(
                f"invalid modal channel split: total={input_channels}, rgb={rgb_channels}, auxiliary={aux_channels}"
            )
        dims, depths, ratios = self._VARIANTS[variant]
        self.rgb_channels = int(rgb_channels)
        self.aux_channels = aux_channels
        self.channel = list(dims)
        total_depth = sum(depths)
        path_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()
        cursor = 0
        self.rgb_patches = nn.ModuleList()
        self.aux_patches = nn.ModuleList()
        self.rgb_stages = nn.ModuleList()
        self.aux_stages = nn.ModuleList()
        for stage, (dim, depth, ratio) in enumerate(zip(dims, depths, ratios)):
            rgb_in = self.rgb_channels if stage == 0 else dims[stage - 1]
            aux_in = self.aux_channels if stage == 0 else dims[stage - 1]
            self.rgb_patches.append(_PatchEmbed(rgb_in, dim, stage == 0))
            self.aux_patches.append(_PatchEmbed(aux_in, dim, stage == 0))
            cross = stage >= 2
            self.rgb_stages.append(
                nn.ModuleList(
                    _DACBlock(dim, ratio, drop_rate, path_rates[cursor + j], cross_modal=cross)
                    for j in range(depth)
                )
            )
            self.aux_stages.append(
                nn.ModuleList(
                    _DACBlock(dim, ratio, drop_rate, path_rates[cursor + j], cross_modal=cross)
                    for j in range(depth)
                )
            )
            cursor += depth
        self.final_fusion = _AttentionFusion(dims[-1])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.rgb_channels + self.aux_channels:
            raise ValueError(
                f"DACFusionBackbone expected BCHW with {self.rgb_channels + self.aux_channels} channels, got {x.shape}"
            )
        rgb, aux = torch.split(x, (self.rgb_channels, self.aux_channels), dim=1)
        outputs = []
        for stage, (rgb_patch, aux_patch, rgb_blocks, aux_blocks) in enumerate(
            zip(self.rgb_patches, self.aux_patches, self.rgb_stages, self.aux_stages)
        ):
            rgb, aux = rgb_patch(rgb), aux_patch(aux)
            if stage < 2:
                for rgb_block, aux_block in zip(rgb_blocks, aux_blocks):
                    rgb, aux = rgb_block(rgb), aux_block(aux)
            else:
                for rgb_block, aux_block in zip(rgb_blocks, aux_blocks):
                    old_rgb, old_aux = rgb, aux
                    rgb = rgb_block(old_rgb, old_aux)
                    aux = aux_block(old_aux, old_rgb)
            outputs.append(self.final_fusion(rgb, aux) if stage == 3 else rgb)
        return outputs


__all__ = ("DACFusionBackbone",)
