# 感谢多模态群友 Accepted 对本文件复现以及分享
# 论文方法：C2DFF-Net / EI2Det / CFGPNet / MCF-Net
# 来源：paper_ports 复现包（本地适配实现）
"""Ports of four open-source dual-modal detectors into the local YOLOMM graph.

The source projects use mutually incompatible YOLO forks.  These modules keep the defining
operations of each method while using the repository-wide ``forward([rgb, auxiliary])`` contract.
They contain no hard-coded device, image size, class count, or absolute path.

Implemented families:
    C2DFF-Net: CDFIM difference interaction and frequency-aware CGSA fusion.
    EI2Det: pooled cross-modal interaction, illumination/quality weighting and edge guidance.
    CFGPNet: RepViT-CSPELAN backbone block and EMA/ASAF-style cross-modal fusion.
    MCF-Net: mixed pixel-difference C2f and grid cross-modal spatial fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conv import Conv


def _pair(x, name):
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} expects [rgb, auxiliary]")
    a, b = x
    if a.ndim != 4 or b.ndim != 4 or a.shape != b.shape:
        raise ValueError(f"{name} requires equal BCHW tensors, got {getattr(a, 'shape', None)} and {getattr(b, 'shape', None)}")
    return a, b


class _CPCAChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        avg = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)), inplace=True)).sigmoid()
        maximum = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1)), inplace=True)).sigmoid()
        return x * (avg + maximum)


class C2DFFCdfim(nn.Module):
    """C2DFF-Net cross-modal difference feature interaction module.

    The learned difference is added back to both streams, matching the source CDFIM + FeatureAdd
    topology while returning two tensors that the YOLOMM parser can route independently.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pre = nn.Conv2d(channels, channels, 1)
        self.ca = _CPCAChannelAttention(channels, reduction)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw7 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels),
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels),
        )
        self.dw11 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels),
            nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels),
        )
        self.dw21 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels),
            nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels),
        )
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        rgb, aux = _pair(x, self.__class__.__name__)
        difference = self.ca(F.gelu(self.pre(rgb - aux)))
        shared = rgb + aux + 2.0 * difference
        base = self.dw5(shared)
        spatial = self.pre(base + self.dw7(base) + self.dw11(base) + self.dw21(base))
        delta = self.out(spatial * shared)
        return rgb + delta, aux + delta


class _C2DFFFrequency(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rgb_pre = nn.Conv2d(channels, channels, 1)
        self.aux_pre = nn.Conv2d(channels, channels, 1)
        self.amp = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.LeakyReLU(0.1), nn.Conv2d(channels, channels, 1))
        self.phase = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.LeakyReLU(0.1), nn.Conv2d(channels, channels, 1))
        self.rgb_post = nn.Conv2d(channels, channels, 1)
        self.aux_post = nn.Conv2d(channels, channels, 1)

    def forward(self, rgb, aux):
        dtype, size = rgb.dtype, rgb.shape[-2:]
        fr = torch.fft.rfft2(self.rgb_pre(rgb).float() + 1e-8)
        fa = torch.fft.rfft2(self.aux_pre(aux).float() + 1e-8)
        amp = self.amp(torch.cat((fr.abs(), fa.abs()), 1))
        phase = self.phase(torch.cat((torch.angle(fr), torch.angle(fa)), 1))
        shared = torch.polar(amp.clamp_min(1e-8), phase)
        spatial = torch.fft.irfft2(shared, s=size).abs().to(dtype)
        return self.rgb_post(spatial - rgb), self.aux_post(spatial - aux)


class _GatedModality(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.left = nn.Conv2d(channels, channels, 1, bias=False)
        self.right = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Conv2d(2 * channels, 1, 1, bias=False)

    def forward(self, left, right):
        z = self.gate(torch.cat((left, right), 1)).sigmoid()
        return z * self.left(left).tanh() + (1.0 - z) * self.right(right).tanh()


class C2DFFCgsa(nn.Module):
    """C2DFF-Net frequency-aware cross-gated channel selective aggregation."""

    def __init__(self, channels, use_frequency=True):
        super().__init__()
        hidden = max(1, channels // 2)
        self.frequency = _C2DFFFrequency(channels) if use_frequency else None
        self.value = nn.Conv2d(channels, hidden, 1)
        self.query = nn.Conv2d(channels, 1, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.rgb_gate = _GatedModality(channels)
        self.aux_gate = _GatedModality(channels)
        self.out = nn.Conv2d(2 * channels, channels, 1)

    def _channel_weight(self, value, query):
        b, c, _, _ = value.shape
        context = value.flatten(2) @ query.flatten(2).softmax(-1).transpose(1, 2)
        context = self.expand(context.unsqueeze(-1)).flatten(2).transpose(1, 2)
        return self.norm(context).transpose(1, 2).reshape(b, -1, 1, 1).sigmoid()

    def forward(self, x):
        rgb, aux = _pair(x, self.__class__.__name__)
        if self.frequency is not None:
            rgb, aux = self.frequency(rgb, aux)
        joint_query = self.query(rgb) + self.query(aux)
        wr = self._channel_weight(self.value(rgb), joint_query)
        wa = self._channel_weight(self.value(aux), joint_query)
        rgb_out = self.rgb_gate(wa * aux, rgb)
        aux_out = self.aux_gate(wr * rgb, aux)
        return self.out(torch.cat((rgb_out, aux_out), 1))


class _CrossTokenAttention(nn.Module):
    def __init__(self, channels, heads=8, dropout=0.0):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = heads
        self.dim = channels // heads
        self.scale = self.dim**-0.5
        self.rgb_qkv = nn.Linear(channels, 3 * channels)
        self.aux_qkv = nn.Linear(channels, 3 * channels)
        self.rgb_out = nn.Linear(channels, channels)
        self.aux_out = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)

    def _split(self, x, layer):
        b, n, c = x.shape
        q, k, v = layer(x).reshape(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, rgb, aux):
        qr, kr, vr = self._split(rgb, self.rgb_qkv)
        qa, ka, va = self._split(aux, self.aux_qkv)
        rgb_attn = self.drop((qa @ kr.transpose(-2, -1) * self.scale).softmax(-1))
        aux_attn = self.drop((qr @ ka.transpose(-2, -1) * self.scale).softmax(-1))
        b, _, n, _ = rgb_attn.shape
        rgb_out = (rgb_attn @ vr).transpose(1, 2).reshape(b, n, -1)
        aux_out = (aux_attn @ va).transpose(1, 2).reshape(b, n, -1)
        return self.rgb_out(rgb_out), self.aux_out(aux_out)


class EI2DetFusion(nn.Module):
    """EI2Det CIM + feature-quality IWM + fixed-edge EFM fusion.

    The original IWM/EFM consume the raw visible image.  In a YAML feature graph the raw image is
    not available at every pyramid level, so the same operations are applied to RGB features.  This
    remains fully differentiable and also permits RGB and Depth to use different source channels.
    """

    def __init__(self, channels, anchors=8, heads=8, layers=1, dropout=0.0, edge_ratio=0.2):
        super().__init__()
        self.grid = (int(anchors), int(anchors))
        self.edge_ratio = float(edge_ratio)
        self.rgb_pos = nn.Parameter(torch.zeros(1, anchors * anchors, channels))
        self.aux_pos = nn.Parameter(torch.zeros(1, anchors * anchors, channels))
        self.attn = nn.ModuleList(_CrossTokenAttention(channels, heads, dropout) for _ in range(layers))
        self.rgb_norm = nn.LayerNorm(channels)
        self.aux_norm = nn.LayerNorm(channels)
        self.rgb_mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels))
        self.aux_mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels))
        quality_hidden = max(8, channels // 8)
        self.quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * channels, quality_hidden, 1), nn.ReLU(),
            nn.Conv2d(quality_hidden, 2, 1), nn.Softmax(dim=1),
        )
        self.edge_project = nn.Conv2d(channels, channels, 1)
        self.out = Conv(2 * channels, channels, 1)
        kernel = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]) / 9.0
        self.register_buffer("edge_kernel", kernel.view(1, 1, 3, 3), persistent=False)

    def forward(self, x):
        rgb, aux = _pair(x, self.__class__.__name__)
        b, c, h, w = rgb.shape
        rp = F.adaptive_avg_pool2d(rgb, self.grid).flatten(2).transpose(1, 2) + self.rgb_pos
        ap = F.adaptive_avg_pool2d(aux, self.grid).flatten(2).transpose(1, 2) + self.aux_pos
        for attention in self.attn:
            rd, ad = attention(self.rgb_norm(rp), self.aux_norm(ap))
            rp = rp + rd
            ap = ap + ad
            rp = rp + self.rgb_mlp(self.rgb_norm(rp))
            ap = ap + self.aux_mlp(self.aux_norm(ap))
        rp = F.interpolate(rp.transpose(1, 2).reshape(b, c, *self.grid), (h, w), mode="bilinear", align_corners=False) + rgb
        ap = F.interpolate(ap.transpose(1, 2).reshape(b, c, *self.grid), (h, w), mode="bilinear", align_corners=False) + aux
        weights = self.quality(torch.cat((rgb, aux), 1))
        fused = self.out(torch.cat((rp * weights[:, :1], ap * weights[:, 1:]), 1))
        gray = rgb.mean(1, keepdim=True)
        edge = F.conv2d(gray, self.edge_kernel.to(gray.dtype), padding=1).abs().expand(-1, c, -1, -1)
        edge = self.edge_project(edge)
        return fused * (1.0 - self.edge_ratio) + edge * self.edge_ratio


class _RepViTDW(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        hidden = max(8, channels // 4)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        y = self.bn(self.dw3(x) + self.dw1(x) + x)
        return y * self.se(y)


class _RepViTBottleneck(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.token = _RepViTDW(channels) if use_se else nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.BatchNorm2d(channels))
        self.channel = nn.Sequential(Conv(channels, 2 * channels, 1), Conv(2 * channels, channels, 1, act=False))

    def forward(self, x):
        x = x + self.token(x)
        return x + self.channel(x)


class CFGPRepViTCSPELAN4(nn.Module):
    """CFGPNet RepViT-CSPELAN block with branch LayerScale and output residual."""

    def __init__(self, c1, c2, n=1, e=0.5, use_se=True, init_gamma=0.01):
        super().__init__()
        hidden = max(8, int(c2 * e))
        self.cv1 = Conv(c1, 2 * hidden, 1)
        self.branch2 = nn.Sequential(*(_RepViTBottleneck(hidden, use_se) for _ in range(n)), Conv(hidden, hidden, 3))
        self.branch3 = nn.Sequential(*(_RepViTBottleneck(hidden, use_se) for _ in range(n)), Conv(hidden, hidden, 3))
        self.cv4 = Conv(4 * hidden, c2, 1)
        self.gamma_branch = nn.Parameter(torch.full((2 * hidden,), float(init_gamma)))
        self.gamma_out = nn.Parameter(torch.full((c2,), float(init_gamma))) if c1 == c2 else None

    def forward(self, x):
        y0, y1 = self.cv1(x).chunk(2, 1)
        y2 = self.branch2(y1)
        y3 = self.branch3(y2)
        y23 = torch.cat((y2, y3), 1) * self.gamma_branch.view(1, -1, 1, 1)
        out = self.cv4(torch.cat((y0, y1, y23), 1))
        return x + self.gamma_out.view(1, -1, 1, 1) * out if self.gamma_out is not None else out


class _EnhancedEMA(nn.Module):
    def __init__(self, channels, groups=16, kernel=7):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups:
            groups -= 1
        self.groups = groups
        cg = channels // groups
        self.axis = nn.Conv1d(cg, cg, kernel, padding=kernel // 2, groups=cg, bias=False)
        self.norm = nn.GroupNorm(cg, cg)
        self.local = nn.Conv2d(cg, cg, 3, padding=1)
        self.tau = nn.Parameter(torch.ones(()))

    def weights(self, x):
        b, c, h, w = x.shape
        g, cg = self.groups, c // self.groups
        z = x.reshape(b * g, cg, h, w)
        zh = torch.sigmoid(self.norm(self.axis(z.mean(3))))[..., None]
        zw = torch.sigmoid(self.norm(self.axis(z.mean(2))))[:, :, None, :]
        first, second = self.norm(z * zh * zw), self.local(z)
        q1 = F.adaptive_avg_pool2d(first, 1).flatten(1).softmax(-1)
        q2 = F.adaptive_avg_pool2d(second, 1).flatten(1).softmax(-1)
        sh = (first.mean(3) * q2[..., None]).sum(1) + (second.mean(3) * q1[..., None]).sum(1)
        sw = (first.mean(2) * q2[..., None]).sum(1) + (second.mean(2) * q1[..., None]).sum(1)
        wh = (sh / self.tau.clamp_min(1e-4)).softmax(-1)[..., None]
        ww = (sw / self.tau.clamp_min(1e-4)).softmax(-1)[:, None, :]
        return torch.bmm(wh, ww).unsqueeze(1)


class CFGPFusion(nn.Module):
    """CFGPNet CEA cross-gating followed by multi-scale ASAF aggregation."""

    def __init__(self, channels, groups=16):
        super().__init__()
        self.ema = _EnhancedEMA(channels, groups)
        hidden = max(8, channels // 2)
        self.reduce = Conv(2 * channels, hidden, 1)
        self.paths = nn.ModuleList((Conv(hidden, hidden, 3), Conv(hidden, hidden, 5), Conv(hidden, hidden, 7)))
        self.select = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3 * hidden, 3, 1), nn.Softmax(dim=1))
        self.out = Conv(hidden, channels, 1)

    def forward(self, x):
        rgb, aux = _pair(x, self.__class__.__name__)
        b, c, h, w = rgb.shape
        g, cg = self.ema.groups, c // self.ema.groups
        wr = self.ema.weights(rgb)
        wa = self.ema.weights(aux)
        rgb = (rgb.reshape(b * g, cg, h, w) * wa.sigmoid()).reshape(b, c, h, w)
        aux = (aux.reshape(b * g, cg, h, w) * wr.sigmoid()).reshape(b, c, h, w)
        base = self.reduce(torch.cat((aux, rgb), 1))
        branches = [path(base) for path in self.paths]
        alpha = self.select(torch.cat(branches, 1))
        fused = sum(branch * alpha[:, i : i + 1] for i, branch in enumerate(branches))
        return self.out(fused)


class _MixedDifferenceConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1):
        super().__init__()
        if k != 3:
            raise ValueError("MCF mixed difference convolution requires k=3")
        self.stride, self.groups = s, g
        self.weight = nn.Parameter(torch.empty(c2, c1 // g, 3, 3))
        self.bias = nn.Parameter(torch.zeros(c2))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.mix = nn.Parameter(torch.ones(4))
        self.alpha = nn.Conv2d(c1, c2, 3, stride=s, padding=1, groups=g)

    def forward(self, x):
        w = self.weight
        vanilla = F.conv2d(x, w, self.bias, self.stride, 1, groups=self.groups)
        center = F.conv2d(x, w.sum((2, 3), keepdim=True), None, self.stride, 0, groups=self.groups)
        central = vanilla - center
        flat = w.flatten(2)
        angular_w = (flat - flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view_as(w)
        angular = F.conv2d(x, angular_w, self.bias, self.stride, 1, groups=self.groups)
        radial_w = w.new_zeros(w.shape[0], w.shape[1], 25)
        radial_w[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, 1:]
        radial_w[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, 1:]
        radial = F.conv2d(x, radial_w.view(w.shape[0], w.shape[1], 5, 5), self.bias, self.stride, 2, groups=self.groups)
        coeff = self.mix.softmax(0)
        difference = coeff[1] * central + coeff[2] * angular + coeff[3] * radial
        gate = self.alpha(x).sigmoid()
        return gate * vanilla + (1.0 - gate) * (coeff[0] * vanilla + difference)


class _PDCBottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.cv1 = nn.Sequential(_MixedDifferenceConv(channels, channels), nn.BatchNorm2d(channels), nn.SiLU())
        self.cv2 = nn.Sequential(_MixedDifferenceConv(channels, channels), nn.BatchNorm2d(channels), nn.SiLU())
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class MCFPDCC2f(nn.Module):
    """MCF-Net difference-convolution C2f backbone block."""

    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True):
        super().__init__()
        self.c = max(8, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.blocks = nn.ModuleList(_PDCBottleneck(self.c, shortcut) for _ in range(n))
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        parts.extend(block(parts[-1]) for block in self.blocks)
        return self.cv2(torch.cat(parts, 1))


class MCFBiFusion(nn.Module):
    """MCF-Net grid cross-modal transform plus multi-kernel spatial fusion."""

    def __init__(self, channels, grid=8, head_dim=16):
        super().__init__()
        self.grid, self.scale = int(grid), head_dim**-0.5
        self.qr, self.kr = nn.Linear(channels, head_dim, bias=False), nn.Linear(channels, head_dim, bias=False)
        self.qa, self.ka = nn.Linear(channels, head_dim, bias=False), nn.Linear(channels, head_dim, bias=False)
        self.vr, self.va = nn.Linear(channels, channels, bias=False), nn.Linear(channels, channels, bias=False)
        self.rgb_gamma, self.aux_gamma = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.rgb_spatial = nn.ModuleList((Conv(channels, 1, 3), Conv(channels, 1, 7)))
        self.aux_spatial = nn.ModuleList((Conv(channels, 1, 3), Conv(channels, 1, 7)))
        self.merge = Conv(2, 1, 5)
        self.rgb_compress, self.aux_compress = Conv(channels, 1, 3), Conv(channels, 1, 3)
        self.out = Conv(2 * channels, channels, 1)

    def forward(self, x):
        rgb, aux = _pair(x, self.__class__.__name__)
        b, c, h, w = rgb.shape
        rt = F.adaptive_avg_pool2d(rgb, self.grid).flatten(2).transpose(1, 2)
        at = F.adaptive_avg_pool2d(aux, self.grid).flatten(2).transpose(1, 2)
        r_attn = (self.qr(rt) @ self.ka(at).transpose(1, 2) * self.scale).softmax(-1)
        a_attn = (self.qa(at) @ self.kr(rt).transpose(1, 2) * self.scale).softmax(-1)
        re = (r_attn @ self.va(at)).transpose(1, 2).reshape(b, c, self.grid, self.grid)
        ae = (a_attn @ self.vr(rt)).transpose(1, 2).reshape(b, c, self.grid, self.grid)
        rgb = rgb + self.rgb_gamma * F.interpolate(re, (h, w), mode="bilinear", align_corners=False)
        aux = aux + self.aux_gamma * F.interpolate(ae, (h, w), mode="bilinear", align_corners=False)
        wr = self.merge(torch.cat([m(rgb) for m in self.rgb_spatial], 1))
        wa = self.merge(torch.cat([m(aux) for m in self.aux_spatial], 1))
        global_map = self.rgb_compress(rgb) + self.aux_compress(aux)
        rgb = rgb * (global_map + wa).sigmoid()
        aux = aux * (global_map + wr).sigmoid()
        return self.out(torch.cat((rgb, aux), 1))


__all__ = (
    "C2DFFCdfim", "C2DFFCgsa", "EI2DetFusion", "CFGPRepViTCSPELAN4", "CFGPFusion",
    "MCFPDCC2f", "MCFBiFusion",
)
