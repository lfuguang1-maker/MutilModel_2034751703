"""GAFF fusion module ported from Iaboni & Abichandani, CVPR 2026.

论文: "Tri-Modal Fusion Transformers for UAV-based Object Detection"（消融算子 GAFF）
来源: Zhang et al., "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection", WACV 2021
链接: https://arxiv.org/abs/2604.16630 / https://github.com/radlab-sketch/trimodal-uav-det

说明:
- 各路 SE 通道重标定 -> 1x1 conv 出双向引导图残差注入 -> concat + 1x1 合并
- 接口: forward(x1, x2) 支持 YAML 双输入 [left, right]，两路等通道等空间尺寸
- 输出单张量 (B, out_channels, H, W)，默认保持通道数不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GAFF']


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: GAP -> FC -> ReLU -> FC -> Sigmoid."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        y = self.gap(x).view(B, C)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).view(B, C, 1, 1)
        return x * y.expand_as(x)


class InterModalityAttention(nn.Module):
    """Concat[x1, x2] -> 1x1 Conv -> 双向引导图 (w1<-2, w2<-1)。"""

    def __init__(self, channels, shared=False):
        super().__init__()
        self.shared = shared
        if shared:
            self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        else:
            self.conv_1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
            self.conv_2 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)
        if self.shared:
            w1, w2 = torch.chunk(self.conv(cat), 2, dim=1)
        else:
            w1, w2 = self.conv_1(cat), self.conv_2(cat)
        return self.sigmoid(w1), self.sigmoid(w2)


class GAFF(nn.Module):
    """Guided Attentive Feature Fusion 融合块。

    Args:
        channels (int): 每路输入通道数（parse_model 自动注入）。
        se_reduction (int): SE 降维比，论文推荐 4 或 8。
        inter_shared (bool): 双向引导是否共享 1x1 conv。
        merge_bottleneck (bool): 合并时走 2C -> C -> 2C bottleneck 路径。
    """

    def __init__(self, channels, se_reduction=4, inter_shared=False, merge_bottleneck=False):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.se_1 = SEBlock(channels, reduction=se_reduction)
        self.se_2 = SEBlock(channels, reduction=se_reduction)
        self.inter_attn = InterModalityAttention(channels, shared=inter_shared)
        self.merge_bottleneck = merge_bottleneck
        if merge_bottleneck:
            self.merge_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
            self.merge_bn1 = nn.BatchNorm2d(channels)
            self.merge_conv2 = nn.Conv2d(channels, self.out_channels, kernel_size=1, bias=False)
            self.merge_bn2 = nn.BatchNorm2d(self.out_channels)
        else:
            self.merge_conv = nn.Conv2d(channels * 2, self.out_channels, kernel_size=1, bias=False)
            self.merge_bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        x1_se = self.se_1(x1)
        x2_se = self.se_2(x2)
        w1, w2 = self.inter_attn(x1, x2)
        x1_guided = x1_se + w1 * x2
        x2_guided = x2_se + w2 * x1
        cat = torch.cat([x1_guided, x2_guided], dim=1)
        if self.merge_bottleneck:
            out = self.merge_bn2(self.merge_conv2(F.relu(self.merge_bn1(self.merge_conv1(cat)), inplace=True)))
        else:
            out = self.merge_bn(self.merge_conv(cat))
        return out

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, merge_bottleneck={self.merge_bottleneck}'
