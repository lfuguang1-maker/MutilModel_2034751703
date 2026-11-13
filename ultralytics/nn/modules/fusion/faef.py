"""
Frequency-Aware Enhancement and Fusion (FAEF).

论文: FAE-Det: Frequency-Aware Enhancement Network for Infrared-Visible Object Detection
来源: IEEE TCSVT 2026 (Early Access), DOI 10.1109/TCSVT.2026.3723123
      https://ieeexplore.ieee.org/document/11653382
官方代码未发布，本实现按论文 Eq.(24)-(39) 复现。

FAEF 接收 CFM 输出的两模态特征，高低频解耦后分频支增强再融合：
1. FDM 频率分解 (Eq.24-26): AvgPool 2x 得低频 [B,C,H/2,W/2]，F - UP(F_low) 得高频
2. 低频支路: VSS Block (SS2D) 全局建模 (Eq.28-29)，用 mobilemamba/vmambanew 的 SS2D
   （自带 torch fallback，无 mamba-ssm 依赖也可 CPU/GPU 运行）
3. 高频支路 HFAE (Eq.30-35): 双尺度 3x3 路径 + sigmoid 门控增强
   F^ = σ(F1) ⊗ F2 + F1 + F2, 其中 F1/F2 为两条 Conv1x1+Conv3x3 路径输出
4. HFCI 高频跨模态交互 (Eq.36-39): CBS 双门控交叉调制
   W_RGB = σ(CBS(F_RGB)) ⊗ CBS(F_IR); F̂_RGB = σ(W_RGB) ⊗ F_IR + F_RGB（对称）
5. 重组（论文公式未完整给出，依 Fig.3/7 结构推断）:
   Out = Conv1x1(Concat(UP(F̂_rgb_low) + F̂_rgb_high, UP(F̂_ir_low) + F̂_ir_high))
   输出单路特征送 Neck。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.mamba.mobilemamba.vmambanew import SS2D

__all__ = ['FAEF']


class _VSSLowFreq(nn.Module):
    """低频支路 VSS Block 等效封装 (Eq.28-29): SS2D 全局建模 + 残差.

    使用 mobilemamba/vmambanew 的 SS2D（自带 torch fallback，无 mamba-ssm 也可运行）；
    channel_first=True / k_group=4 / forward_type='v05' 为其 BCHW 输入的可运行组合。
    """

    def __init__(self, ch: int):
        super().__init__()
        self.ss2d = SS2D(d_model=ch, d_state=16, ssm_ratio=1, initialize="v2",
                         forward_type="v05", channel_first=True, k_group=4)

    def forward(self, x):
        return x + self.ss2d(x)


class _FDM(nn.Module):
    """频率分解模块 (Eq.24-26): 低频 = AvgPool 2x 下采样，高频 = 输入 - 上采样低频."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        low = self.pool(x)
        high = x - F.interpolate(low, size=x.shape[-2:], mode='nearest')
        return low, high


class _HFAE(nn.Module):
    """高频自适应增强 (Eq.30-35): 双尺度卷积路径 + sigmoid 门控残差."""

    def __init__(self, ch: int):
        super().__init__()
        self.path1 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 1))
        self.path2 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        f1 = self.path1(x)
        f2 = self.path2(x)
        return torch.sigmoid(f1) * f2 + f1 + f2  # Eq.32/35


class _CBS(nn.Module):
    """CBS: 1x1 Conv + BN + SiLU (FAE-Det 论文记号)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _HFCI(nn.Module):
    """高频跨模态交互 (Eq.36-39): 级联双门控粗筛→细调."""

    def __init__(self, ch: int):
        super().__init__()
        self.cbs_rgb = _CBS(ch)
        self.cbs_ir = _CBS(ch)

    def forward(self, f_rgb, f_ir):
        w_rgb = torch.sigmoid(self.cbs_rgb(f_rgb)) * self.cbs_ir(f_ir)   # Eq.36
        f_rgb = torch.sigmoid(w_rgb) * f_ir + f_rgb                      # Eq.37
        w_ir = torch.sigmoid(self.cbs_ir(f_ir)) * self.cbs_rgb(f_rgb)    # Eq.38
        f_ir = torch.sigmoid(w_ir) * f_rgb + f_ir                        # Eq.39
        return f_rgb, f_ir


class FAEF(nn.Module):
    """频率感知增强与融合模块 (FAE-Det, Eq.24-39).

    Args:
        in_channels (int): 输入通道数（两路需一致）。
        out_channels (int): 输出通道数，默认等于 in_channels（拼接后 1x1 压缩）。
    Returns:
        torch.Tensor: 融合后单路特征 [B, out_channels, H, W]。
    """

    def __init__(self, in_channels: int, out_channels: int = 0):
        super().__init__()
        ouc = out_channels or in_channels
        self.fdm = _FDM()
        self.vss_rgb = _VSSLowFreq(in_channels)
        self.vss_ir = _VSSLowFreq(in_channels)
        self.hfae_rgb = _HFAE(in_channels)
        self.hfae_ir = _HFAE(in_channels)
        self.hfci = _HFCI(in_channels)
        self.fuse = nn.Conv2d(in_channels * 2, ouc, 1)

    def forward(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            raise ValueError(f"FAEF expects 2 inputs (rgb, ir), got {type(x)}")
        f_rgb, f_ir = x
        # (1) 频率分解 (Eq.24-26)
        low_rgb, high_rgb = self.fdm(f_rgb)
        low_ir, high_ir = self.fdm(f_ir)
        # (2) 低频全局建模 (Eq.28-29)
        low_rgb = self.vss_rgb(low_rgb)
        low_ir = self.vss_ir(low_ir)
        # (3) 高频增强 (Eq.30-35)
        high_rgb = self.hfae_rgb(high_rgb)
        high_ir = self.hfae_ir(high_ir)
        # (4) 高频跨模态交互 (Eq.36-39)
        high_rgb, high_ir = self.hfci(high_rgb, high_ir)
        # (5) 重组（推断项，见文件头注释）
        out_rgb = high_rgb + F.interpolate(low_rgb, size=high_rgb.shape[-2:], mode='nearest')
        out_ir = high_ir + F.interpolate(low_ir, size=high_ir.shape[-2:], mode='nearest')
        return self.fuse(torch.cat([out_rgb, out_ir], dim=1))
