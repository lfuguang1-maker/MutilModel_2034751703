"""
Cross-Modal Spatial Attention Fusion (CSAF).

论文: FAE-Det: Frequency-Aware Enhancement Network for Infrared-Visible Object Detection
来源: IEEE TCSVT 2026 (Early Access), DOI 10.1109/TCSVT.2026.3723123
      https://ieeexplore.ieee.org/document/11653382
官方代码未发布（LJs-Projects/FAE-Det 暂为空仓库），本实现按论文 Eq.(4)-(7) 逐式复现。

CSAF 是轻量骨干级渐进跨模态交互模块（论文嵌入骨干第 2-4 尺度）：
1. 每个模态经 7x7 depthwise conv + BN + Sigmoid 生成单通道空间注意力图 (Eq.4-5)
2. 用对方模态的注意力图交叉调制当前模态并残差连接 (Eq.6-7):
   F_IR^  = F_IR  + F_IR  ⊗ A_RGB
   F_RGB^ = F_RGB + F_RGB ⊗ A_IR
输出为两路增强特征 (F_RGB^, F_IR^)，通道与输入一致，供双路骨干继续独立提取。
"""

import torch
import torch.nn as nn

__all__ = ['CSAF']


class _SpatialAttentionCSAF(nn.Module):
    """CSAF 空间注意力: 7x7 depthwise conv + BN + Sigmoid, 输出单通道权重图 (Eq.4-5)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, 7, padding=3, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        return torch.sigmoid(self.bn(self.conv(x)))


class CSAF(nn.Module):
    """跨模态空间注意力融合模块 (FAE-Det, Eq.4-7).

    Args:
        in_channels (int): 输入通道数（两路需一致）。
        out_channels (int): 输出通道数（保持与输入一致，仅作签名兼容）。
    Returns:
        tuple(torch.Tensor): (F_rgb_enhanced, F_ir_enhanced)，形状均为 [B, C, H, W]。
    """

    def __init__(self, in_channels: int, out_channels: int = 0):
        super().__init__()
        if out_channels and out_channels != in_channels:
            raise ValueError(f"CSAF requires out_channels == in_channels, got {in_channels} vs {out_channels}")
        self.attn_rgb = _SpatialAttentionCSAF(in_channels)
        self.attn_ir = _SpatialAttentionCSAF(in_channels)

    def forward(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            raise ValueError(f"CSAF expects 2 inputs (rgb, ir), got {type(x)}")
        f_rgb, f_ir = x
        a_rgb = self.attn_rgb(f_rgb)  # A_RGB (Eq.5)
        a_ir = self.attn_ir(f_ir)     # A_IR  (Eq.4)
        # 交叉调制 + 残差 (Eq.6-7)
        f_ir = f_ir + f_ir * a_rgb
        f_rgb = f_rgb + f_rgb * a_ir
        return f_rgb, f_ir
