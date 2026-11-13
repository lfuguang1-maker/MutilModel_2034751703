"""Tri-modal fusion operators ported from Iaboni & Abichandani, CVPR 2026.

论文: "Tri-Modal Fusion Transformers for UAV-based Object Detection"
链接: https://arxiv.org/abs/2604.16630
代码: https://github.com/radlab-sketch/trimodal-uav-det (trimodaldet/models/fusion.py)

说明:
- MAGEBiTE = MAGE(跨模态通道+空间门控残差) + BiTE(双向 token 交换 + DWConv/1x1 精炼)
- 接口: forward(x1, x2) 支持 YAML 双输入 [left, right]，两路等通道等空间尺寸
- 输出单张量 (B, C, H, W)，保持 stride 与通道不变，即插即用
"""

import math

import torch
import torch.nn as nn

__all__ = ['MAGE', 'BiTEBlock', 'MAGEBiTE']


class MAGEChannelGate(nn.Module):
    """Channel attention weights for inter-modal feature rectification."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4 // reduction, dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B = x1.shape[0]
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        mx = self.max_pool(x).view(B, self.dim * 2)
        y = self.mlp(torch.cat((avg, mx), dim=1)).view(B, self.dim * 2, 1)
        # -> (2, B, C, 1, 1)：索引 0 作用于 x2，索引 1 作用于 x1
        return y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class MAGESpatialGate(nn.Module):
    """Spatial attention weights for inter-modal feature rectification."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # (2, B, 1, H, W)


class MAGE(nn.Module):
    """Modality-Aware Gated Exchange: 双向通道+空间门控的跨模态残差整流。

    Args:
        dim (int): 输入通道数。
        reduction (int): 注意力 MLP 降维比。
        lambda_c (float): 通道门权重。
        lambda_s (float): 空间门权重。
    """

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = MAGEChannelGate(dim=dim, reduction=reduction)
        self.spatial_weights = MAGESpatialGate(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2
        out_x2 = x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1
        return out_x1, out_x2


class InterModalAttention(nn.Module):
    """Linear-style bidirectional inter-modal attention (official implementation)."""

    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        return x1, x2


class CrossPath(nn.Module):
    """Cross-path module combining channel projection and inter-modal attention."""

    def __init__(self, dim, reduction=1, num_heads=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = InterModalAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        out_x1 = self.norm1(x1 + self.end_proj1(torch.cat((y1, v1), dim=-1)))
        out_x2 = self.norm2(x2 + self.end_proj2(torch.cat((y2, v2), dim=-1)))
        return out_x1, out_x2


class MAGEChannelEmbed(nn.Module):
    """Channel embedding module with residual connection (2C -> C)."""

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        return self.norm(self.residual(x) + self.channel_embed(x))


class BiTEBlock(nn.Module):
    """Bidirectional Token Exchange: 双向 token 交换后 2C -> C 合并。

    Args:
        dim (int): 每路输入通道数。
        reduction (int): 内部降维比。
        num_heads (int): 交叉注意力头数。
    """

    def __init__(self, dim, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = MAGEChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                            norm_layer=norm_layer)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        return self.channel_emb(torch.cat((x1, x2), dim=-1), H, W)


class MAGEBiTE(nn.Module):
    """MAGE + BiTE 完整融合块（论文 baseline fusion block，即插即用版）。

    MAGE 先对两路特征做双向通道/空间门控残差整流，BiTE 再做双向 token
    交换并合并为单张量，输出保持输入 stride 与通道数不变。

    Args:
        channels (int): 每路输入通道数（parse_model 自动注入）。
        reduction (int): 内部降维比。
        num_heads (int): 交叉注意力头数。
        lambda_c (float): MAGE 通道门权重。
        lambda_s (float): MAGE 空间门权重。
    """

    def __init__(self, channels, reduction=1, num_heads=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.mage = MAGE(dim=channels, reduction=reduction, lambda_c=lambda_c, lambda_s=lambda_s)
        self.bite = BiTEBlock(dim=channels, reduction=reduction, num_heads=num_heads)

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        x1, x2 = self.mage(x1, x2)
        return self.bite(x1, x2)
