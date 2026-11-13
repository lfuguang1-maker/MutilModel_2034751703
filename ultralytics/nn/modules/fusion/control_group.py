# 感谢多模态群友 Accepted 对本文件复现以及分享
# 论文方法：CFT / EAEFNet / Fusion-Mamba / MMI-Det
# 来源：paper_ports 复现包（本地适配实现）
"""Runnable ports of dual-modal fusion blocks from the local ``COntrol Group`` archives.

The original projects target different YOLO generations and use incompatible model parsers.  The
classes in this file preserve their characteristic interaction math while exposing the common
``forward([rgb, auxiliary])`` interface used by this repository.

Source families:
    - EAEFNet: explicit correlation and complementary channel/spatial fusion.
    - CFT: pooled-token self-attention over concatenated RGB/auxiliary tokens.
    - MMI-Det: frequency/pattern enhanced pooled-token interaction.
    - Fusion-Mamba: channel exchange followed by intra- and cross-modal spatial mixing.

ICAFusion already has a native port in :mod:`icafusion`; DACFusion is a complete backbone and is
therefore implemented in :mod:`ultralytics.nn.backbone.dacfusion`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_pair(x, module_name: str):
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{module_name} expects [rgb, auxiliary], got {type(x).__name__}")
    rgb, aux = x
    if rgb.ndim != 4 or aux.ndim != 4:
        raise ValueError(f"{module_name} expects two BCHW tensors")
    if rgb.shape != aux.shape:
        raise ValueError(f"{module_name} requires equal feature shapes, got {rgb.shape} and {aux.shape}")
    return rgb, aux


class PairAdd(nn.Module):
    """Element-wise addition used by the original CFT/Fusion-Mamba/MMI-Det pyramids."""

    def __init__(self, channels: int | None = None):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        left, right = _validate_pair(x, self.__class__.__name__)
        return left + right


class _SelfAttention(nn.Module):
    """CFT-style multi-head self-attention."""

    def __init__(self, dim: int, heads: int, attn_drop: float, proj_drop: float):
        super().__init__()
        if dim % heads:
            raise ValueError(f"channels ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class _TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: float, attn_drop: float, resid_drop: float):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, heads, attn_drop, resid_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(resid_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(resid_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PooledTokenInteraction(nn.Module):
    """Shared implementation for the CFT and MMI-Det interaction families."""

    def __init__(
        self,
        channels: int,
        vert_anchors: int = 8,
        horz_anchors: int = 8,
        heads: int = 8,
        expansion: float = 4.0,
        layers: int = 8,
        emb_drop: float = 0.1,
        attn_drop: float = 0.1,
        resid_drop: float = 0.1,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels = channels
        self.grid = (int(vert_anchors), int(horz_anchors))
        token_count = 2 * self.grid[0] * self.grid[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, channels))
        self.embed_drop = nn.Dropout(emb_drop)
        self.blocks = nn.Sequential(
            *[_TransformerLayer(channels, heads, expansion, attn_drop, resid_drop) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(channels)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _prepare(self, rgb: torch.Tensor, aux: torch.Tensor):
        return F.adaptive_avg_pool2d(rgb, self.grid), F.adaptive_avg_pool2d(aux, self.grid)

    def _interact(self, rgb: torch.Tensor, aux: torch.Tensor):
        b, c, _, _ = rgb.shape
        pooled_rgb, pooled_aux = self._prepare(rgb, aux)
        rgb_tokens = pooled_rgb.flatten(2).transpose(1, 2)
        aux_tokens = pooled_aux.flatten(2).transpose(1, 2)
        tokens = torch.cat((rgb_tokens, aux_tokens), dim=1)
        tokens = self.norm(self.blocks(self.embed_drop(tokens + self.pos_embed)))
        split = self.grid[0] * self.grid[1]
        rgb_out, aux_out = tokens[:, :split], tokens[:, split:]
        rgb_out = rgb_out.transpose(1, 2).reshape(b, c, *self.grid)
        aux_out = aux_out.transpose(1, 2).reshape(b, c, *self.grid)
        size = rgb.shape[-2:]
        rgb_out = F.interpolate(rgb_out, size=size, mode="bilinear", align_corners=False)
        aux_out = F.interpolate(aux_out, size=size, mode="bilinear", align_corners=False)
        return rgb_out, aux_out


class CFTInteraction(_PooledTokenInteraction):
    """CFT GPT block plus the two source ``Add2`` residual operations.

    Returning both enhanced streams lets later backbone stages continue to learn modality-specific
    features, exactly as in the source three-stage CFT topology.
    """

    def forward(self, x):
        rgb, aux = _validate_pair(x, self.__class__.__name__)
        rgb_delta, aux_delta = self._interact(rgb, aux)
        return rgb + rgb_delta, aux + aux_delta


class _PatternExtractor(nn.Module):
    def __init__(self, channels: int, patterns: int = 8):
        super().__init__()
        self.patterns = patterns
        self.mask = nn.Sequential(
            nn.Conv2d(channels, patterns, 1, bias=False),
            nn.BatchNorm2d(patterns),
            nn.SiLU(),
            nn.Conv2d(patterns, patterns, 3, padding=1, groups=patterns),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(patterns, channels, 1, bias=False)

    def forward(self, x: torch.Tensor):
        masks = self.mask(x)
        return x * self.project(masks).sigmoid(), masks


class MMIDetInteraction(_PooledTokenInteraction):
    """Executable MMI-Det interaction with corrected real-valued Fourier enhancement.

    The supplied MMI-Det code applies ``Conv2d`` to complex FFT tensors and computes a non-scalar
    separation matrix that is never added to its trainer loss.  Here the FFT magnitude is real, and
    ``pattern_loss`` is a normalized scalar retained for inspection or an explicit future loss hook.
    Detection loss behaviour is deliberately left unchanged.
    """

    def __init__(
        self,
        channels: int,
        vert_anchors: int = 8,
        horz_anchors: int = 8,
        heads: int = 8,
        expansion: float = 4.0,
        layers: int = 8,
        emb_drop: float = 0.1,
        attn_drop: float = 0.1,
        resid_drop: float = 0.1,
        frequency: bool = False,
        patterns: int = 8,
    ):
        super().__init__(
            channels, vert_anchors, horz_anchors, heads, expansion, layers, emb_drop, attn_drop, resid_drop
        )
        self.frequency = bool(frequency)
        self.rgb_patterns = _PatternExtractor(channels, patterns)
        self.aux_patterns = _PatternExtractor(channels, patterns)
        self.frequency_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.pattern_loss = torch.tensor(0.0)

    @staticmethod
    def _high_frequency(x: torch.Tensor) -> torch.Tensor:
        # A smooth high-pass residual is stable, differentiable and remains real-valued.
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        return (x - low).abs()

    @staticmethod
    def _separation_loss(*masks: torch.Tensor) -> torch.Tensor:
        losses = []
        for mask in masks:
            vectors = F.normalize(mask.flatten(2), dim=-1)
            gram = vectors @ vectors.transpose(-1, -2)
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
            losses.append(((gram - eye) ** 2).mean())
        return torch.stack(losses).mean()

    def _prepare(self, rgb: torch.Tensor, aux: torch.Tensor):
        rgb = F.adaptive_avg_pool2d(rgb, self.grid)
        aux = F.adaptive_avg_pool2d(aux, self.grid)
        rgb_pattern, rgb_masks = self.rgb_patterns(rgb)
        aux_pattern, aux_masks = self.aux_patterns(aux)
        if self.frequency:
            rgb_hf, aux_hf = self._high_frequency(rgb), self._high_frequency(aux)
            gate = self.frequency_gate(torch.cat((rgb_hf, aux_hf), dim=1))
            rgb_pattern = rgb_pattern + gate * rgb_hf
            aux_pattern = aux_pattern + (1.0 - gate) * aux_hf
        self.pattern_loss = self._separation_loss(rgb_masks, aux_masks)
        return rgb + rgb_pattern, aux + aux_pattern

    def forward(self, x):
        rgb, aux = _validate_pair(x, self.__class__.__name__)
        rgb_delta, aux_delta = self._interact(rgb, aux)
        return rgb + rgb_delta, aux + aux_delta


class _ConvMambaFallback(nn.Module):
    """Pure-PyTorch spatial state mixer used when selective-scan CUDA is unavailable."""

    def __init__(self, channels: int, expansion: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, 1)
        self.scan_h = nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden)
        self.scan_v = nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        self.drop = nn.Dropout2d(drop)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        value = self.scan_h(value) + self.scan_v(value)
        value = F.silu(value) * torch.sigmoid(gate)
        return x + self.scale * self.drop(self.out_proj(value))


class FusionMambaInteraction(nn.Module):
    """Fusion-Mamba channel swapping and symmetric cross-modal spatial mixing.

    This port removes the source's hard-coded ``.cuda()`` allocation.  It keeps the defining
    channel-quarter exchange and bidirectional mixing while using a portable PyTorch spatial scan,
    so YAML parsing, CPU model construction and multi-GPU workers all behave consistently.
    """

    def __init__(self, channels: int, depth: int = 1, expansion: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.channels = channels
        self.rgb_blocks = nn.ModuleList(_ConvMambaFallback(channels, expansion, drop) for _ in range(depth))
        self.aux_blocks = nn.ModuleList(_ConvMambaFallback(channels, expansion, drop) for _ in range(depth))
        self.rgb_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.aux_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.rgb_cross = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.aux_cross = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.rgb_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))
        self.aux_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    @staticmethod
    def _swap_quarters(rgb: torch.Tensor, aux: torch.Tensor):
        # tensor_split also supports widths not divisible by four, unlike the source implementation.
        rgb_parts = list(torch.tensor_split(rgb, 4, dim=1))
        aux_parts = list(torch.tensor_split(aux, 4, dim=1))
        rgb_parts[1], aux_parts[1] = aux_parts[1], rgb_parts[1]
        rgb_parts[3], aux_parts[3] = aux_parts[3], rgb_parts[3]
        return torch.cat(rgb_parts, dim=1), torch.cat(aux_parts, dim=1)

    def forward(self, x):
        rgb, aux = _validate_pair(x, self.__class__.__name__)
        rgb, aux = self._swap_quarters(rgb, aux)
        for rgb_block, aux_block in zip(self.rgb_blocks, self.aux_blocks):
            rgb, aux = rgb_block(rgb), aux_block(aux)
        joint = torch.cat((rgb, aux), dim=1)
        rgb_out = rgb + self.rgb_scale * self.rgb_gate(joint) * self.rgb_cross(aux)
        aux_out = aux + self.aux_scale * self.aux_gate(joint) * self.aux_cross(rgb)
        return rgb_out, aux_out


class _EAEFFeaturePool(nn.Module):
    def __init__(self, channels: int, ratio: int = 2):
        super().__init__()
        self.down = nn.Linear(channels, channels * ratio)
        self.up = nn.Linear(channels * ratio, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.up(F.gelu(self.down(pooled)))


class _EAEFChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.down = nn.Linear(channels, hidden)
        self.up = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.up(F.gelu(self.down(pooled))).unsqueeze(-1).unsqueeze(-1)


class EAEFInteraction(nn.Module):
    """Explicit Attention-Enhanced Fusion used by EAEFNet's dual YOLOv5 backbone.

    The module returns ``(enhanced_rgb, enhanced_auxiliary, fused)``.  The first two outputs feed
    the next modality-specific stage and the third is retained for the detection pyramid.  This
    mirrors ``BiYOLOv5CSPDarknet`` while replacing its hard-coded CUDA tensor with ``ones_like``.
    """

    def __init__(self, channels: int, pool_ratio: int = 2, attention_ratio: int = 16):
        super().__init__()
        self.channels = channels
        self.pool = _EAEFFeaturePool(channels, pool_ratio)
        # Source EAEF uses groups=C for a 2C-channel convolution (two channels per group).
        self.explicit_context = nn.Conv2d(channels * 2, channels * 2, 7, padding=3, groups=channels)
        self.channel_attention = _EAEFChannelAttention(channels * 2, attention_ratio)
        self.rgb_spatial = nn.Conv2d(channels, 1, 1)
        self.aux_spatial = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        rgb, aux = _validate_pair(x, self.__class__.__name__)
        b, c, _, _ = rgb.shape
        rgb_descriptor = F.normalize(self.pool(rgb), dim=1, eps=1e-6)
        aux_descriptor = F.normalize(self.pool(aux), dim=1, eps=1e-6)
        correlation = c * rgb_descriptor.unsqueeze(2) @ aux_descriptor.unsqueeze(1)
        cross_gate = torch.diagonal(torch.sigmoid(correlation), dim1=-2, dim2=-1).view(b, c, 1, 1)
        complementary_gate = torch.ones_like(cross_gate) - cross_gate

        rgb_explicit, aux_explicit = rgb * cross_gate, aux * cross_gate
        rgb_complementary, aux_complementary = rgb * complementary_gate, aux * complementary_gate
        explicit = torch.cat((rgb_explicit, aux_explicit), dim=1)
        modal_gate = torch.sigmoid(self.channel_attention(self.explicit_context(explicit)))
        rgb_gate, aux_gate = modal_gate.split(c, dim=1)
        enhanced_rgb = rgb_explicit * rgb_gate + rgb_complementary
        enhanced_aux = aux_explicit * aux_gate + aux_complementary

        spatial = torch.softmax(
            torch.cat((self.rgb_spatial(enhanced_rgb), self.aux_spatial(enhanced_aux)), dim=1), dim=1
        )
        fused = enhanced_rgb * spatial[:, :1] + enhanced_aux * spatial[:, 1:]
        return enhanced_rgb, enhanced_aux, fused


__all__ = (
    "PairAdd",
    "CFTInteraction",
    "MMIDetInteraction",
    "FusionMambaInteraction",
    "EAEFInteraction",
)
