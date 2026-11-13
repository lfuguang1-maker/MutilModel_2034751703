# Multimodal subset sampling & correlated failure modes (CVPR 方案 §3.4)
# 7 非空子集枚举、课程式概率、相关故障模式、batch 全模态硬保证。

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence

__all__ = ["SUBSETS", "CORRELATED_FAILURES", "curriculum_probs", "sample_subset", "sample_batch",
           "correlated_failure_subset", "subset_avail"]

_LETTERS = ["R", "E", "T", "D"]  # 槽位 0/1/2/3 的子集标签字母（D=Depth）


def _enumerate_subsets(n_modalities: int) -> List[str]:
    """n 模态的全部非空子集标签（'R','E','T','RE',...；第 4 模态 D=Depth）。"""
    letters = ["R", "E", "T", "D"][:n_modalities]
    return ["".join(s) for r in range(1, n_modalities + 1) for s in itertools.combinations(letters, r)]


SUBSETS: Dict[int, List[str]] = {n: _enumerate_subsets(n) for n in (1, 2, 3, 4)}

# 相关故障模式 → 存活槽位集合（以槽位索引表示）；None 表示参数化模式
CORRELATED_FAILURES = {
    "appearance_fail": (0, 2),  # RGB+Thermal 同失效（外观组），仅 3 模态有效
    "motion_fail": (1,),        # Event 失效（运动组）
    "single_fail": None,        # 单传感器故障：keep=存活槽位元组
    "all_but_one": None,        # 只保留一个模态：keep=(idx,)
}


def subset_avail(subset: str, n_modalities: int = 3) -> List[float]:
    """子集标签 → mm_avail 向量：'RT' → [1,0,1]。"""
    return [1.0 if _LETTERS[i] in subset else 0.0 for i in range(n_modalities)]


def correlated_failure_subset(mode: str, keep: Optional[Sequence[int]] = None, n_modalities: int = 3) -> str:
    """相关故障模式 → 存活子集标签。

    appearance_fail 需 n_modalities>=3（RGB/Event/Thermal 槽位 0/1/2）。
    """
    if mode == "appearance_fail":
        if n_modalities < 3:
            raise ValueError("appearance_fail 需要 >=3 模态（RGB+Event+Thermal）")
        return "E"
    if mode == "motion_fail":
        return "".join(_LETTERS[i] for i in range(n_modalities) if i != 1)
    if mode == "single_fail":
        if keep is None:
            raise ValueError("single_fail 需要 keep 参数（存活槽位索引）")
        return "".join(_LETTERS[i] for i in keep)
    if mode == "all_but_one":
        if keep is None or len(keep) != 1:
            raise ValueError("all_but_one 需要 keep=(idx,) 单槽位")
        return _LETTERS[keep[0]]
    raise ValueError(f"未知相关故障模式: {mode}")


def curriculum_probs(epoch_frac: float, n_modalities: int = 3) -> Dict[str, float]:
    """子集课程概率（构建方案 §5.3）。

    epoch 0-10%: 全模态 0.90；10-50%: 0.50；50%+: 0.30；其余均分给其他子集。
    返回 dict 概率和为 1。
    """
    frac = min(max(float(epoch_frac), 0.0), 1.0)
    subsets = SUBSETS[n_modalities]
    full = subsets[-1]  # 'RET'（全部槽位）
    if frac < 0.10:
        p_full, rest = 0.90, 0.10
    elif frac < 0.50:
        p_full, rest = 0.50, 0.50
    else:
        p_full, rest = 0.30, 0.70
    others = [s for s in subsets if s != full]
    probs = {s: rest / len(others) for s in others}
    probs[full] = p_full
    return probs


def sample_subset(rng: random.Random, probs: Dict[str, float]) -> str:
    names = list(probs)
    return rng.choices(names, weights=[probs[n] for n in names])[0]


def sample_batch(rng: random.Random, probs: Dict[str, float], batch_size: int = 16) -> List[str]:
    """采样一个 batch 的子集标签；若无全模态样本则显式替换第一个（构建方案 §5.3 硬保证）。"""
    batch = [sample_subset(rng, probs) for _ in range(batch_size)]
    full = sorted(probs, key=lambda s: -len(s))[0]  # 最长子集 = 全模态
    if full not in batch:
        batch[0] = full
    return batch
