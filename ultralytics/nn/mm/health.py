# Multimodal Health-Utility Fusion Modules (CVPR 方案 §4)
# MMMask / MMHealthHead(原子化) / MMUtilityHead / MMGroupRouter / MMModalityRouter
# MMNullToken / MMSharedExpert / MMHealthFusion
#
# 语义：
# - health：测量质量（数据可不可信），K 个可学习退化原子（SD-CR 借鉴），rho = 1 - Σ p_k·s_k
# - utility：任务相关性，由单模态辅助损失目标 exp(-L_aux/tau) 监督
# - 路由：两级 softmax（组级 -> 组内模态级），权重在可用模态上归一化

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MMMask", "MMHealthHead", "MMUtilityHead", "MMGroupRouter", "MMModalityRouter",
    "MMNullToken", "MMSharedExpert", "MMHealthFusion", "MMAuxDetectHead",
]


class MMMask(nn.Identity):
    """Router 'Mask' 伪源的占位模块：输入即 [B,M,H,W] availability mask，恒等直通。"""

    def __init__(self, c1: int = 0):
        super().__init__()
        self.c1 = c1
        self.c2 = c1


class MMHealthHead(nn.Module):
    """原子化健康头：K 个可学习退化原子 + 可学习严重度，输出 rho ∈ [0,1] 与原子权重。

    输入特征统计 [GAP, GMP, std] + availability a_m。
    rho = 1 - Σ softmax(net(h))_k · softmax(severity)_k（构建方案 §4.2）。
    """

    def __init__(self, c: int, hidden: int = 64, k: int = 8):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(c * 3 + 1, hidden), nn.ReLU(), nn.Linear(hidden, k))
        self.severity = nn.Parameter(torch.full((k,), 0.5))

    def forward(self, f: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gap = f.mean((2, 3))
        gmp = F.adaptive_max_pool2d(f, 1).flatten(1)
        std = f.flatten(2).std(-1)
        h = torch.cat([gap, gmp, std, a.unsqueeze(1)], 1)
        logits = self.net(h)                                           # [B,K]
        self.last_logits = logits                                      # 类型监督用（logits）
        p = torch.softmax(logits, 1)                                   # 原子权重（概率单纯形）
        rho = 1 - (p * torch.softmax(self.severity, 0)).sum(1)         # [B]
        return rho.clamp(0, 1), p


class MMUtilityHead(nn.Module):
    """效用头：任务相关性 u_m ∈ [0,1]，目标来自单模态辅助损失（L_util = MSE(u, exp(-L_aux/tau))）。"""

    def __init__(self, c: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = torch.cat([f.mean((2, 3)), f.flatten(2).std(-1)], 1)
        return torch.sigmoid(self.net(h)).squeeze(1)


class MMGroupRouter(nn.Module):
    """组级路由：输入各组 [feat_stat, group_health, group_avail]，输出组权重 softmax。"""

    def __init__(self, c_out: int, groups: Sequence[Sequence[int]]):
        super().__init__()
        self.groups = tuple(tuple(g) for g in groups)
        self.proj = nn.Linear(3, 16)

    def forward(self, feats: List[torch.Tensor], h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        scores = []
        for g in self.groups:
            gfeat = sum(feats[i] for i in g) / len(g)
            gstat = gfeat.mean((2, 3)).mean(1)      # [B]
            gh = h[:, g].mean(1) if hasattr(h, "shape") and h.dim() == 2 else h[list(g)].mean()
            ga = a[:, list(g)].max(1).values        # [B]
            scores.append(self.proj(torch.stack([gstat, gh, ga], 1)).mean(-1))  # [B]（proj 输出 16 维取均值）
        return torch.softmax(torch.stack(scores, 1), 1)  # [B,G]


class MMModalityRouter(nn.Module):
    """组内模态路由：输入 [rho, u, a, feat_stat]，输出各模态 score（组内 softmax 由调用方完成）。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, feats: List[torch.Tensor], h: torch.Tensor, u: torch.Tensor, a: torch.Tensor,
                group: Sequence[int]) -> torch.Tensor:
        scores = []
        for i in group:
            stat = feats[i].mean((2, 3)).mean(1)    # [B]
            inp = torch.stack([h[:, i], u[:, i], a[:, i], stat], 1)
            scores.append(self.proj(inp).squeeze(-1))
        return torch.stack(scores, 1)               # [B,len(group)]


class MMNullToken(nn.Parameter):
    """缺失槽位特征替换：每模态可学习 [C,1,1]。以 Parameter 形式由 MMHealthFusion 持有。"""


class MMSharedExpert(nn.Module):
    """始终在线共享分支：DWConv3x3 + PWConv1x1 + BN + SiLU；全缺失时作为唯一输出兜底。"""

    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MMHealthFusion(nn.Module):
    """健康-效用融合层（构建方案 §4.2）。

    输入 xs = [f_0, ..., f_{M-1}, mask]，mask 为 [B,M,H,W]。
    输出融合特征 [B,c_out,H,W]；缓存 last_health/last_utility/last_weights/last_feats 供损失使用。
    """

    def __init__(self, c_in_list: Sequence[int], c_out: int, groups: Sequence[Sequence[int]] = ((0, 1), (2,)),
                 use_shared: bool = True, health_hidden: int = 64, utility_hidden: int = 64, k_atoms: int = 8):
        super().__init__()
        self.M = len(c_in_list) - 1
        self.groups = tuple(tuple(g) for g in groups)
        self.proj = nn.ModuleList([nn.Conv2d(c, c_out, 1) for c in c_in_list[:-1]])
        self.health = nn.ModuleList([MMHealthHead(c_out, health_hidden, k=k_atoms) for _ in range(self.M)])
        self.utility = nn.ModuleList([MMUtilityHead(c_out, utility_hidden) for _ in range(self.M)])
        self.group_router = MMGroupRouter(c_out, self.groups)
        self.mod_router = MMModalityRouter()
        self.null = nn.Parameter(torch.zeros(self.M, c_out, 1, 1))
        nn.init.normal_(self.null, std=0.02)
        self.shared = MMSharedExpert(c_out) if use_shared else None
        self.fuse = nn.Conv2d(c_out, c_out, 1)
        # 训练期缓存（每 step 清理，避免保留计算图）
        self.last_health = self.last_utility = self.last_weights = None
        self.last_atoms = None
        self.last_feats = None

    def route(self, feats, h, u, a):
        """两级 softmax：组级 × 组内模态级；缺失置零后在可用模态上重归一化。"""
        B = feats[0].shape[0]
        w = torch.zeros(B, self.M, device=feats[0].device, dtype=feats[0].dtype)
        w_g = self.group_router(feats, h, a)                    # [B,G]
        for gi, g in enumerate(self.groups):
            scores = self.mod_router(feats, h, u, a, g)         # [B,len(g)]
            wm = torch.softmax(scores, 1)
            for j, i in enumerate(g):
                w[:, i] = w_g[:, gi] * wm[:, j]
        w = w * a
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        return w

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
            xs = xs[0]
        feats = [p(x) for p, x in zip(self.proj, xs[:-1])]
        mask = xs[-1]                                          # [B,M,H,W]
        if mask.dtype != feats[0].dtype:
            mask = mask.to(feats[0].dtype)
        B = feats[0].shape[0]
        a = F.adaptive_avg_pool2d(mask, 1).flatten(1)          # [B,M]
        h_out = [self.health[i](feats[i], a[:, i]) for i in range(self.M)]
        rho = torch.stack([o[0] for o in h_out], 1)            # [B,M]
        atoms = torch.stack([self.health[i].last_logits for i in range(self.M)], 1)  # [B,M,K] 原子 logits
        u = torch.stack([self.utility[i](feats[i]) for i in range(self.M)], 1)
        w = self.route(feats, rho, u, a)
        feats = [a[:, i].view(B, 1, 1, 1) * feats[i] + (1 - a[:, i].view(B, 1, 1, 1)) * self.null[i]
                 for i in range(self.M)]
        fused = sum(w[:, i].view(B, 1, 1, 1) * feats[i] for i in range(self.M))
        if self.shared is not None:
            fused = fused + self.shared(fused)
        self.last_health, self.last_utility, self.last_weights = rho, u, w
        self.last_atoms = atoms
        self.last_feats = feats
        return self.fuse(fused)


class MMAuxDetectHead(nn.Module):
    """共享辅助检测头（构建方案 §4.3）：每模态每尺度一个 1x1 卷积，复用主检测损失。

    输入为各 MMHealthFusion 缓存的「融合前、按模态投影后」特征，全部 detach，
    因此梯度只进入辅助头本身，不回流主干与融合层。
    """

    def __init__(self, chs: Sequence[int], nc: int, reg_max: int):
        super().__init__()
        self.no = 4 * int(reg_max) + int(nc)
        self.convs = nn.ModuleList([nn.Conv2d(int(c), self.no, 1) for c in chs])

    def forward(self, feats_per_scale, batch, criterion, avail=None):
        """feats_per_scale: [S][M] → 每模态检测损失 [B,M]（逐样本相同，按模态聚合）。"""
        if not feats_per_scale or not feats_per_scale[0]:
            return None
        B = feats_per_scale[0][0].shape[0]
        M = len(feats_per_scale[0])
        losses = []
        for m in range(M):
            preds = [conv(feats[m].detach()) for conv, feats in zip(self.convs, feats_per_scale)]
            loss_vec, _ = criterion(preds, batch)
            losses.append(loss_vec.sum() / max(B, 1))
        return torch.stack(losses, 0).unsqueeze(0).expand(B, M)


class MMHealthDetectionModelMixin:
    """健康-效用检测模型的 loss 扩展（构建方案 §5.1）。

    total = L_det + λ_aux·L_aux + λ_health·L_health + λ_util·L_util + λ_bal·L_bal
    - L_aux: 各 MMHealthFusion 的 last_feats 送共享辅助头（由 trainer 注入 self.mm_aux_head）
    - L_health: rho 与 mm_corruption（原子化：type CE + 强度回归在 head 内部，这里回归 rho 标量）
    - L_util: u vs exp(-L_aux_m/tau)，显式掩码均值
    - L_bal: 路由权重负熵
    """

    LAMBDA_AUX = 0.5
    LAMBDA_HEALTH = 0.1
    LAMBDA_UTIL = 0.1
    LAMBDA_BAL = 0.01
    TYPE_CE_WEIGHT = 0.5
    UTIL_TAU = 1.0

    def _concat_mask(self, batch):
        """[B,fused_ch,H,W] + [B,M,H,W] mask → [B,input_ch,H,W]（collate 后拼接，避免增广污染）。"""
        img = batch["img"]
        avail = batch.get("mm_avail")  # [B,M]
        if avail is None:
            return img
        router = getattr(self, "mm_router", None) or getattr(self, "multimodal_router", None)
        if router is None or router.contract.mask_channels <= 0:
            return img
        if img.shape[1] == router.contract.input_ch:
            return img  # 已拼接
        if not torch.is_tensor(avail):
            avail = torch.as_tensor(np.asarray(avail), dtype=torch.float32)
        B, _, H, W = img.shape
        mask = avail.to(img.device, img.dtype)[:, :, None, None].expand(-1, -1, H, W)
        return torch.cat([img, mask], dim=1)

    def _collect_health_stats(self):
        """收集所有 MMHealthFusion 层的缓存统计（rho / 原子 logits / u / 路由权重）。"""
        rhos, atoms, us, ws = [], [], [], []
        for m in self.model.modules() if hasattr(self, "model") else self.modules():
            if isinstance(m, MMHealthFusion) and m.last_weights is not None:
                rhos.append(m.last_health)
                atoms.append(m.last_atoms)
                us.append(m.last_utility)
                ws.append(m.last_weights)
        return rhos, atoms, us, ws

    def _clear_health_cache(self):
        for m in self.model.modules() if hasattr(self, "model") else self.modules():
            if isinstance(m, MMHealthFusion):
                m.last_health = m.last_utility = m.last_weights = m.last_feats = None
                m.last_atoms = None
                for head in m.health:
                    head.last_logits = None

    def health_loss(self, batch):
        """L_health：可用模态上的 rho 回归 + 退化类型交叉熵（构建方案 §4.2）。

        - rho: 无退化 target=1，退化 target=1-level；仅 avail=1 的模态计入
        - type CE: 8 个原子对应 8 类退化，仅「可用且确有退化」的模态计入
        """
        rhos, atoms, _, _ = self._collect_health_stats()
        corr = batch.get("mm_corruption")  # [B,M,2] 或 list
        if not rhos or corr is None:
            return torch.zeros((), device=next(self.parameters()).device)
        rho = torch.stack(rhos, 0).mean(0)  # [B,M]（尺度平均）
        if isinstance(corr, (list, tuple)):
            corr = torch.stack([torch.as_tensor(c) for c in corr], 0)
        corr = corr.to(rho.device, rho.dtype)
        level = corr[..., 1]                # [B,M] 归一化强度
        target = (1 - level).clamp(0, 1)    # 无退化 target=1
        avail = batch.get("mm_avail")
        avail = torch.ones_like(rho) if avail is None else avail.to(rho.device, rho.dtype)
        mse = (((rho - target) ** 2) * avail).sum() / avail.sum().clamp_min(1.0)

        ce = torch.zeros((), device=rho.device)
        sel_mask = (avail > 0) & (level > 0)
        if sel_mask.any() and atoms and atoms[0] is not None:
            logits = torch.stack(atoms, 0).mean(0)          # [B,M,K]
            k = logits.shape[-1]
            cls = (corr[..., 0] * k).round().long() - 1     # 0=无退化 → -1
            sel = sel_mask & (cls >= 0) & (cls < k)
            if sel.any():
                ce = F.cross_entropy(logits[sel], cls[sel])
        return mse + self.TYPE_CE_WEIGHT * ce

    def util_loss(self, aux_per_modality, avail):
        """L_util：u 逼近按可用模态相对排序的效用目标（构建方案 §5.1）。

        target = exp(-(L_m - min_m L) / tau)：最优模态取 1，其余按损失差衰减。
        用相对差而非绝对损失，避免辅助头初始损失量级（可达 1e2）让 exp(-L) 全为 0、
        效用头被压成常数 0。avail=0 的模态不计入。
        """
        _, _, us, _ = self._collect_health_stats()
        if not us:
            return torch.zeros((), device=next(self.parameters()).device)
        u = torch.stack(us, 0).mean(0)      # [B,M]
        tau = float(getattr(self, "mm_util_tau", self.UTIL_TAU) or self.UTIL_TAU)
        avail = avail.to(u.device, u.dtype)
        loss_m = aux_per_modality.detach()
        # 仅在可用模态上取最小值作为基准，避免被掉线槽位的零值干扰
        big = torch.finfo(loss_m.dtype).max
        base = torch.where(avail > 0, loss_m, torch.full_like(loss_m, big)).min(1, keepdim=True).values
        base = torch.where(torch.isfinite(base), base, torch.zeros_like(base))
        # 无可用模态的样本（理论上被硬保证排除）用 0 占位，避免 inf*0=nan；该行随后被 avail 屏蔽
        diff = torch.where(avail > 0, (loss_m - base).clamp_min(0), torch.zeros_like(loss_m))
        target = torch.exp(-diff / max(tau, 1e-6))
        sq = (u - target) ** 2 * avail
        return sq.sum() / avail.sum().clamp_min(1.0)

    def balance_loss(self):
        """L_bal：路由权重负熵（最小化 = 最大化使用熵，防塌缩）。"""
        _, _, _, ws = self._collect_health_stats()
        if not ws:
            return torch.zeros((), device=next(self.parameters()).device)
        w = torch.stack(ws, 0).mean(0)      # [B,M]
        eps = 1e-8
        ent = -(w * (w + eps).log()).sum(-1).mean()
        return -ent

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        x = self._concat_mask(batch)
        preds = self.forward(x) if preds is None else preds
        det_loss_vec, det_items = self.criterion(preds, batch)

        # 辅助损失（trainer 注入 mm_aux_head 时启用）
        aux_loss = None
        avail = batch.get("mm_avail")
        aux_head = getattr(self, "mm_aux_head", None)
        if aux_head is not None and avail is not None:
            feats_per_scale = [m.last_feats for m in self._health_fusions() if m.last_feats is not None]
            aux_loss = aux_head(feats_per_scale, batch, self.criterion, avail)  # [B,M] 每模态检测损失

        health = self.health_loss(batch)
        if aux_loss is not None:
            util = self.util_loss(aux_loss, avail)
            avail_f = avail.to(aux_loss.device, aux_loss.dtype)
            aux_raw = (aux_loss * avail_f).sum() / avail_f.sum().clamp_min(1.0)
            # 辅助头随机初始化时其损失可达 1e2 量级；直接乘 LAMBDA_AUX 会压垮 AMP 的 loss scale。
            # 用自归一化保持 O(LAMBDA_AUX) 的稳定量级，梯度方向不变（等价自适应学习率）。
            total_extra = self.LAMBDA_AUX * (aux_raw / aux_raw.detach().clamp_min(1e-6))
        else:
            util = torch.zeros((), device=det_loss_vec.device)
            aux_raw = torch.zeros((), device=det_loss_vec.device)
            total_extra = torch.zeros((), device=det_loss_vec.device)
        bal = self.balance_loss()
        extra = total_extra + self.LAMBDA_HEALTH * health + self.LAMBDA_UTIL * util + self.LAMBDA_BAL * bal
        self._record_aux_stats(health, util, bal, aux_raw)
        self._clear_health_cache()
        # det_loss_vec 为 [box, cls, dfl]（criterion 已乘 batch_size）；extra 为标量。
        # 历史实现 det_loss_vec + extra 会广播到每一项，等价于 extra 计 3 次且不随 batch_size 缩放。
        # 默认保持该量级（legacy，不改变现有训练行为）；mm_extra_batch_scale=True 时改为设计值。
        out = det_loss_vec.clone()
        n_terms = out.numel() if out.dim() else 1
        scaled = extra * n_terms
        if bool(getattr(self, "mm_extra_batch_scale", False)):
            scaled = scaled * batch["img"].shape[0]
        if out.dim():
            out[0] = out[0] + scaled
        else:
            out = out + scaled
        return out, det_items

    def _record_aux_stats(self, health, util, bal, aux):
        """按 step 缓存分项损失与评分/路由统计（detach），供 trainer 在 epoch 末汇总。"""
        acc = getattr(self, "_mm_aux_acc", None)
        if acc is None:
            acc = self._mm_aux_acc = {}
        for k, v in (("health", health), ("util", util), ("bal", bal), ("aux", aux)):
            if torch.is_tensor(v):
                acc.setdefault(k, []).append(v.detach().reshape(()))
        # 各模态平均：健康度 rho / 效用 u / 路由权重 w（先尺度平均再样本平均）
        rhos, _, us, ws = self._collect_health_stats()
        for key, vals in (("rho", rhos), ("u", us), ("w", ws)):
            if vals and vals[0] is not None:
                acc.setdefault(key, []).append(torch.stack(vals, 0).mean(0).mean(0).detach())

    def _health_fusions(self):
        mods = self.model.modules() if hasattr(self, "model") else self.modules()
        return [m for m in mods if isinstance(m, MMHealthFusion)]
