"""CSSA fusion module ported from Iaboni & Abichandani, CVPR 2026.

论文: "Tri-Modal Fusion Transformers for UAV-based Object Detection"（消融算子 CSSA）
来源: Cao et al., "Multimodal Object Detection by Channel Switching and Spatial Attention", CVPR 2023 PBVS
链接: https://arxiv.org/abs/2604.16630 / https://github.com/radlab-sketch/trimodal-uav-det

说明:
- ECA 通道评分 -> 低置信通道与对方模态同位通道硬替换 -> 均值+最大空间门融合
- 接口: forward(x1, x2) 支持 YAML 双输入 [left, right]，两路等通道等空间尺寸
- 输出单张量 (B, C, H, W)，极轻量即插即用
"""

import torch
import torch.nn as nn

__all__ = ['CSSAFusion']


class ECABlock(nn.Module):
    """Efficient Channel Attention: GAP + 1D conv on channel axis + sigmoid."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv1d(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return self.sigmoid(y)


class ChannelSwitching(nn.Module):
    """低置信通道（w < k）与对方模态同位通道硬替换。"""

    def __init__(self, switching_thresh=0.5):
        super().__init__()
        self.k = switching_thresh

    def forward(self, x, x_prime, w):
        return torch.where(w < self.k, x_prime, x)


class SpatialAttention(nn.Module):
    """均值+最大双空间门作用于 concat 特征后取平均融合。"""

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        C = x1.shape[1]
        x_cat = torch.cat((x1, x2), dim=1)
        w_avg = self.sigmoid(torch.mean(x_cat, dim=1, keepdim=True))
        w_max = self.sigmoid(torch.max(x_cat, dim=1, keepdim=True)[0])
        x_cat_w = x_cat * w_avg * w_max
        return (x_cat_w[:, :C] + x_cat_w[:, C:]) / 2


class CSSAFusion(nn.Module):
    """CSSA (Channel Switching and Spatial Attention) 轻量融合块。

    Args:
        channels (int): 每路输入通道数（parse_model 自动注入）。
        switching_thresh (float): 通道替换阈值 tau，论文最佳 0.5。
        kernel_size (int): ECA 1D 卷积核大小。
    """

    def __init__(self, channels, switching_thresh=0.5, kernel_size=3):
        super().__init__()
        self.eca_1 = ECABlock(kernel_size=kernel_size)
        self.eca_2 = ECABlock(kernel_size=kernel_size)
        self.cs = ChannelSwitching(switching_thresh=switching_thresh)
        self.sa = SpatialAttention()

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        w1 = self.eca_1(x1)
        w2 = self.eca_2(x2)
        # 两路交换均基于原始输入（与官方实现一致）
        y1 = self.cs(x1, x2, w1)
        y2 = self.cs(x2, x1, w2)
        return self.sa(y1, y2)
