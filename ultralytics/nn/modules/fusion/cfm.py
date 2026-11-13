"""
Cross-Modal Frequency Modulation (CFM).

论文: FAE-Det: Frequency-Aware Enhancement Network for Infrared-Visible Object Detection
来源: IEEE TCSVT 2026 (Early Access), DOI 10.1109/TCSVT.2026.3723123
官方代码未发布，本实现按论文 Eq.(8)-(23) 逐式复现。

CFM 在频域对齐 RGB/IR 特征（论文置于骨干后、FAEF 前的两阶段融合第一阶段）：
- 两模态各经 1x1 conv 压缩到 32 通道 (Eq.12-13)
- (a) RGB 高频增强: H=Conv3x3, M=KN(H), F̄ += F̄ - CARAFE(F̄, M)  残差去噪 (Eq.14-15)
- (b) IR 低频引导: L=Conv5x5, L_Fusion = L_IR + CARAFE(L_RGB, KN(L_IR))  (Eq.16-18)
- (c) 低频引导高频: H_Fusion = CARAFE(H_IR, KN(L_Fusion)) + H_RGB  (Eq.19-20)
- (d) 终重构: F_IR_out = CARAFE(F̄_IR, KN(L_Fusion));
              F_RGB_out = F̄_RGB + F̄_RGB - CARAFE(F̄_RGB, KN(H_Fusion))  (Eq.21-23)

KN (Kernel Normalizer, Eq.8-10): 对 K^2 核位置 softmax 归一化（含可选 Hamming 平滑后再归一化）。
CARAFE 此处为核重组版 (Eq.11): 输入 unfold 成 [B,C,K^2,H,W] 与自适应核逐元素乘后沿核维求和，
同分辨率内容感知重组，非上采样版（区别于 ultralytics/nn/sample/carafe.py）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CFM', 'KernelNormalizer', 'CARAFEReassembly']


class KernelNormalizer(nn.Module):
    """核归一化 (Eq.8-10): softmax over K^2 + 可选 Hamming 平滑 + 再归一化.

    Args:
        kernel_size (int): 重组核边长 K，核位置数 K^2.
        hamming (bool): 是否施加 Hamming 窗平滑.
    """

    def __init__(self, kernel_size: int = 3, hamming: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.hamming = hamming

    def forward(self, x):
        # x: [B, C, H, W] -> [B, C, K^2, H, W]
        x = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2).reshape(
            x.shape[0], x.shape[1], self.kernel_size ** 2, x.shape[2], x.shape[3])
        m = torch.softmax(x, dim=2)  # Eq.8
        if self.hamming:
            window = torch.hamming_window(self.kernel_size, periodic=False, device=x.device, dtype=x.dtype)
            window = (window[:, None] * window[None, :]).reshape(1, 1, -1, 1, 1)
            m = m * window  # Eq.9
            m = m / m.sum(dim=2, keepdim=True).clamp_min(1e-8)  # Eq.10
        return m


class CARAFEReassembly(nn.Module):
    """CARAFE 核重组 (Eq.11): 用自适应核权重对 unfold 特征加权聚合，同分辨率输出.

    Args:
        kernel_size (int): 重组核边长 K.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x, kernel):
        # x: [B, C, H, W]; kernel: [B, C, K^2, H, W]
        xf = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2).reshape(
            x.shape[0], x.shape[1], self.kernel_size ** 2, x.shape[2], x.shape[3])
        return (xf * kernel).sum(dim=2)  # Eq.11


class CFM(nn.Module):
    """跨模态频率调制模块 (FAE-Det, Eq.12-23).

    Args:
        in_channels (int): 输入通道数（两路需一致）。
        compress (int): 频域处理压缩通道数（论文 32）。
        kernel_size (int): KN/CARAFE 重组核边长。
    Returns:
        tuple(torch.Tensor): (F_rgb_out, F_ir_out)，形状均为 [B, in_channels, H, W]。
    """

    def __init__(self, in_channels: int, compress: int = 32, kernel_size: int = 3):
        super().__init__()
        c = compress
        self.fcc_rgb = nn.Conv2d(in_channels, c, 1)   # Eq.12
        self.fcc_ir = nn.Conv2d(in_channels, c, 1)    # Eq.13
        self.hfe = nn.Conv2d(c, c, 3, padding=1)      # 高频编码器
        self.lfe = nn.Conv2d(c, c, 5, padding=2)      # 低频编码器
        self.kn = KernelNormalizer(kernel_size)
        self.carafe = CARAFEReassembly(kernel_size)
        # Eq.22-23 终重构在压缩域输出，1x1 还原到原通道宽度（论文声明输出与输入同形）
        self.expand_rgb = nn.Conv2d(c, in_channels, 1)
        self.expand_ir = nn.Conv2d(c, in_channels, 1)

    def forward(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            raise ValueError(f"CFM expects 2 inputs (rgb, ir), got {type(x)}")
        f_rgb, f_ir = x
        f_rgb = self.fcc_rgb(f_rgb)  # Eq.12
        f_ir = self.fcc_ir(f_ir)     # Eq.13
        # (a) RGB 高频增强 (Eq.14-15)
        h_rgb = self.hfe(f_rgb)
        f_rgb = f_rgb + f_rgb - self.carafe(f_rgb, self.kn(h_rgb))
        # (b) IR 低频引导 (Eq.16-18)
        l_rgb, l_ir = self.lfe(f_rgb), self.lfe(f_ir)
        l_fusion = l_ir + self.carafe(l_rgb, self.kn(l_ir))
        # (c) 低频引导高频 (Eq.19-20)
        h_ir = self.hfe(f_ir)
        h_fusion = self.carafe(h_ir, self.kn(l_fusion)) + h_rgb
        # (d) 终重构 (Eq.21-23)
        f_ir_out = self.expand_ir(self.carafe(f_ir, self.kn(l_fusion)))                   # Eq.22
        f_rgb_out = self.expand_rgb(f_rgb + f_rgb - self.carafe(f_rgb, self.kn(h_fusion)))  # Eq.23
        return f_rgb_out, f_ir_out
