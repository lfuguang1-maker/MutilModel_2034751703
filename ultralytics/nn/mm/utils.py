# Ultralytics Multimodal Utilities
# Helper functions for multimodal system status and validation
# Version: v1.0

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from ultralytics.utils import LOGGER

MM_METADATA_KEY = "metadata"
MM_X_MODALITY_KEY = "x_modality_name"
SINGLE_MODAL_BUILD_KEY = "single_modal_build"

# ---------------------------------------------------------------------------
# Input-source alias table: legacy tokens → 槽位身份（A/B/Fuse）
# ---------------------------------------------------------------------------
# 槽位是唯一身份（决策 §1）；RGB/X/Dual、PRIMARY/SECONDARY/FUSED 只做输入别名，
# 解析后立刻落到槽位，内部不再出现角色值。
INPUT_SOURCE_ALIASES: dict[str, str] = {
    "RGB":     "A",
    "X":       "B",
    "PRIMARY": "A",
    "SECONDARY": "B",
    "FUSED":   "Fuse",
    "Dual":    "Fuse",
}

_FUSE_TOKENS = {"fuse", "fused", "dual"}
_SLOT_LETTERS = frozenset(chr(65 + i) for i in range(26))

_CASE_INSENSITIVE_ALIAS_MAP: dict[str, str] = {
    k.casefold(): v for k, v in INPUT_SOURCE_ALIASES.items()
}

_VISIBLE_SYNONYMS = {"rgb", "visible", "vis", "color"}
_IR_SYNONYMS = {"ir", "thermal", "infrared", "tir", "flir"}

@dataclass(frozen=True)
class MultiModalChannelContract:
    """N 模态通道契约 — 槽位是唯一身份，契约是唯一通道真相（决策 §1/§3）。

    Attributes:
        modality_channels: 各槽位通道数元组，如 (3, 3, 3)
        modality_names: 各槽位模态名称元组，如 ('rgb', 'event', 'thermal')
        slots: 各槽位字母标记元组，如 ('A', 'B', 'C')
        is_multimodal_model: 是否为多模态模型
    """

    modality_channels: tuple[int, ...]
    modality_names: tuple[str, ...]
    slots: tuple[str, ...]
    is_multimodal_model: bool
    mask_channels: int = 0  # availability mask 通道数（mm_use_mask=true 时 = 模态数；旧 ckpt 缺省 0）
    fill_kinds: tuple[str, ...] = ()  # 每槽位填充语义：'visible' | 'zero'；缺省按可见光同义词推导

    def __post_init__(self):
        if not self.fill_kinds:
            kinds = tuple(
                "visible" if _normalize_modality_lookup_token(n) in _VISIBLE_SYNONYMS else "zero"
                for n in self.modality_names
            )
            object.__setattr__(self, "fill_kinds", kinds)

    def fill_kind(self, data: Optional[dict] = None, name: Optional[str] = None, slot: Optional[str] = None) -> str:
        """查询槽位填充语义；data.yaml ``modality_fill`` 可按模态名覆盖（决策 §7 Wave 3.3）。"""
        if slot is not None and name is None:
            name = self.modality_names[self.slots.index(slot)]
        idx = next((i for i, n in enumerate(self.modality_names)
                    if _normalize_modality_lookup_token(n) == _normalize_modality_lookup_token(name)), 0)
        override = None
        if isinstance(data, dict):
            mf = data.get("modality_fill")
            if isinstance(mf, dict):
                for k, v in mf.items():
                    if _normalize_modality_lookup_token(k) == _normalize_modality_lookup_token(name):
                        override = str(v)
        if override in ("visible", "zero"):
            return override
        return self.fill_kinds[idx]
    @property
    def fused_ch(self) -> int:
        """融合所有槽位的总通道数。"""
        return sum(self.modality_channels)

    @property
    def input_ch(self) -> int:
        """模型实际输入通道数 = fused_ch + mask_channels。"""
        return self.fused_ch + self.mask_channels

    @property
    def primary_ch(self) -> int:
        """主模态（槽位 A）通道数，只读兼容属性；迁移完成后删除。"""
        return self.modality_channels[0] if self.modality_channels else 0

    @property
    def secondary_ch(self) -> int:
        """次模态（槽位 B）通道数，只读兼容属性；迁移完成后删除。"""
        return self.modality_channels[1] if len(self.modality_channels) > 1 else 0

    @property
    def slices(self) -> tuple[slice, ...]:
        """派生逐槽位通道切片。

        Example:
            对于 (3, 3, 3) 通道数，返回 (slice(0,3), slice(3,6), slice(6,9))
        """
        offsets = [0] + list(itertools.accumulate(self.modality_channels))
        return tuple(slice(offsets[i], offsets[i + 1]) for i in range(len(self.modality_channels)))

    def slot_of(self, name: str) -> str:
        """按模态名查槽位；未配置的名字抛 ValueError。"""
        for slot, mod_name in zip(self.slots, self.modality_names):
            if _normalize_modality_lookup_token(mod_name) == _normalize_modality_lookup_token(name):
                return slot
        raise ValueError(f"模态名 {name!r} 不在契约中: {list(self.modality_names)}")

    def channels_of(self, slot: str) -> int:
        """按槽位查通道数；'Fuse'/'Mask' 为伪源，组合槽（'AB'）返回拼接通道数。"""
        if slot == "Fuse":
            return self.fused_ch
        if slot == "Mask":
            return self.mask_channels
        try:
            return sum(self.modality_channels[self.slots.index(c)] for c in slot)
        except ValueError:
            raise ValueError(f"槽位 {slot!r} 不在契约槽位 {list(self.slots)} 中") from None

    def slice_of(self, slot: str) -> slice:
        """按槽位查切片；组合槽返回跨段切片。"""
        try:
            idx = self.slots.index(slot)
        except ValueError:
            raise ValueError(f"槽位 {slot!r} 不在契约槽位 {list(self.slots)} 中") from None
        return self.slices[idx]

    def to_meta(self) -> dict[str, Any]:
        """序列化为 ckpt / cfg_dict 传递用的最小元数据。"""
        return {
            "modality_used": list(self.modality_names),
            "modality_ch": dict(zip(self.modality_names, self.modality_channels)),
            "mask_channels": self.mask_channels,
        }

    @classmethod
    def from_meta(cls, meta: Optional[dict], is_multimodal_model: bool = True) -> "MultiModalChannelContract":
        """从 ckpt / cfg_dict 元数据还原契约。

        旧 ckpt 只有 ``Xch`` 时按 N==2 解释（slots=(A,B)，channels=(3,Xch)）。
        旧元数据缺 ``mask_channels`` 时默认 0（向后兼容）。
        """
        if not isinstance(meta, dict):
            raise ValueError("契约元数据缺失或格式错误")
        modality_used = meta.get("modality_used")
        modality_ch = meta.get("modality_ch")
        if isinstance(modality_used, (list, tuple)) and isinstance(modality_ch, dict):
            names = [str(n) for n in modality_used]
            channels = tuple(int(modality_ch[n]) for n in names)
        elif "Xch" in meta:
            names = ["rgb", str(meta.get("x_modality_name", "x"))]
            channels = (int(meta.get("Pch", 3)), int(meta["Xch"]))
        else:
            raise ValueError(f"无法从元数据还原通道契约: {meta!r}")
        slots = tuple(resolve_modality_slots(names))
        return cls(
            modality_channels=channels,
            modality_names=tuple(names),
            slots=slots,
            is_multimodal_model=is_multimodal_model,
            mask_channels=int(meta.get("mask_channels", 0) or 0),
        )


@dataclass(frozen=True)
class SingleModalBuildSpec:
    """Build-time routing contract for ``single_modal`` YAML injection."""

    raw: str
    role_token: str
    role_name: str
    matched_modality: Optional[str] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "role_token": self.role_token,
            "role_name": self.role_name,
            "matched_modality": self.matched_modality,
        }


def resolve_input_source_slot(token: object, slots: Optional[Iterable[str]] = None) -> Optional[str]:
    """解析输入源 token 到槽位身份（A/B/C/.../Fuse），一处实现（决策 §7 Wave 2.1）。

    规则：
    - 单个大写/小写字母且属于槽位字母表 → 同名槽位（'a'→'A'）
    - ``'RGB'/'PRIMARY'`` → ``'A'``；``'X'/'SECONDARY'`` → ``'B'``
    - ``'Fuse'/'Dual'/'FUSED'`` → ``'Fuse'``
    - 组合 token（AB/BC/ABD...）：每个字母必须是槽位字母，返回原大写组合
    - 给定 ``slots``（当前契约槽位）时，解析结果必须在其中，否则 ``ValueError``
    - 无法识别 → ``ValueError``；token 非字符串返回 ``None``

    Returns:
        槽位身份字符串；``None`` 仅当 token 不是字符串。
    """
    if not isinstance(token, str):
        return None
    t = token.strip()
    if not t:
        return None
    folded = t.casefold()
    if folded in _CASE_INSENSITIVE_ALIAS_MAP:
        resolved: Optional[str] = _CASE_INSENSITIVE_ALIAS_MAP[folded]
    elif folded in _FUSE_TOKENS:
        resolved = "Fuse"
    elif t.upper() in _SLOT_LETTERS or (len(t) > 1 and all(c.upper() in _SLOT_LETTERS for c in t)):
        resolved = t.upper()
    else:
        raise ValueError(f"无法识别的多模态输入源 token: {token!r}（支持槽位 A-Z、别名 RGB/X/Dual/A/B/Fuse 或组合如 AB）")

    if slots is not None:
        allowed = set(slots) | {"Fuse"}
        if resolved not in allowed and not (len(resolved) > 1 and all(c in set(slots) for c in resolved)):
            raise ValueError(f"输入源 token {token!r} 解析为 {resolved!r}，不在当前契约槽位 {list(slots)} 中")
    return resolved


def resolve_modality_slots(modality_used: list[str]) -> list[str]:
    """将模态列表转为槽位字母序列（A/B/C/...）。
    Args:
        modality_used: 模态名称列表，如 ['rgb', 'event', 'thermal']
    
    Returns:
        槽位字母列表，如 ['A', 'B', 'C']
    
    Example:
        >>> resolve_modality_slots(['rgb', 'event', 'thermal'])
        ['A', 'B', 'C']
    """
    return [chr(65 + i) for i in range(len(modality_used))]  # A=65


def auto_detect_all_modality_channels(
    data: dict,
    modality_used: list[str]
) -> tuple[int, ...]:
    """按 modality_used 顺序自动检测各槽位通道数。
    
    Args:
        data: data.yaml 解析后的字典，包含 'path' 和 'modality' 字段
        modality_used: 模态名称列表，如 ['rgb', 'event', 'thermal']
    
    Returns:
        各槽位通道数元组，如 (3, 3, 3)
    
    Example:
        >>> data = {'path': 'datasets/tri', 'modality': {'rgb': 'images', 'event': 'images_event'}}
        >>> auto_detect_all_modality_channels(data, ['rgb', 'event'])
        (3, 3)
    """
    from ultralytics.data.dataset import auto_detect_modality_channels
    
    dataset_root = Path(data.get('path', '.'))
    channels = []
    for mod_name in modality_used:
        # 统一走 get_modality_path_from_data：兼容 modality/modalities 新旧字段与推断回退
        mod_dir = get_modality_path_from_data(data, mod_name)
        ch = auto_detect_modality_channels(dataset_root, mod_dir)
        channels.append(ch)
    return tuple(channels)


def is_rgb_modality(modality_name: str) -> bool:
    """Return True when *modality_name* refers to RGB (case-insensitive).

    Used to gate colour-space operations (BGR->RGB, HSV augment, etc.).
    Only the exact token ``"rgb"`` (case-insensitive) qualifies.
    """
    return isinstance(modality_name, str) and modality_name.strip().lower() == "rgb"


# Compatibility alias for the accidental public-API rename in the trimodal rewrite.
is_rgb_modality_deprecated = is_rgb_modality


def _clean_modality_value(value: Any) -> Optional[str]:
    """Return a stripped modality token when the input is a non-empty string."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_modality_lookup_token(value: Any) -> Optional[str]:
    """Normalize a modality token for case-insensitive lookup only."""
    cleaned = _clean_modality_value(value)
    return cleaned.casefold() if cleaned else None


def is_rgb_modality_token(value: Any) -> bool:
    """Return True when *value* semantically refers to RGB regardless of case."""
    return _normalize_modality_lookup_token(value) == 'rgb'


def _coerce_configured_modalities(values: Any) -> list[str]:
    """Normalize configured modality containers while preserving configured token spelling."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    cleaned: list[str] = []
    for value in values:
        token = _clean_modality_value(value)
        if token:
            cleaned.append(token)
    return cleaned


_WARNED_DEP_MM_FIELDS: set = set()


def warn_deprecated_mm_fields(data, family=""):
    """检测旧版多模态字段(models/modalities)并 WARNING 提示迁移，按 data path 去重一次。

    新版默认字段为 ``modality``(路径映射 dict) + ``modality_used``(模态组合 list)；
    旧版 ``models`` / ``modalities`` 仍兼容，但建议迁移。同一数据集(path)在一次进程内只提示一次。
    """
    if not isinstance(data, dict):
        return
    dep_key = str(data.get("path") or id(data))  # 同一数据集只提示一次
    if dep_key in _WARNED_DEP_MM_FIELDS:
        return
    found = []
    if "models" in data:
        found.append("'models' → 'modality_used'")
    if "modalities" in data:
        found.append("'modalities' → 'modality'")
    if found:
        prefix = f"[{family}] " if family else ""
        LOGGER.warning(
            f"{prefix}检测到旧版多模态字段，建议迁移到新字段以保持未来兼容: {', '.join(found)}"
        )
        _WARNED_DEP_MM_FIELDS.add(dep_key)


def get_configured_modality_tokens(data: Optional[dict]) -> list[str]:
    """Return configured modality tokens from data.yaml while preserving configured spelling."""
    data = data or {}
    warn_deprecated_mm_fields(data)

    for key in ("modality_used", "models", "modalities"):
        tokens = _coerce_configured_modalities(data.get(key))
        if tokens:
            return tokens

    mapping = data.get("modality") or data.get("modalities")
    if isinstance(mapping, dict):
        tokens = [_clean_modality_value(key) for key in mapping.keys()]
        return [token for token in tokens if token]

    return []


def find_configured_modality_token(token: Any, configured_tokens: Sequence[str]) -> Optional[str]:
    """Find the configured token that matches *token* case-insensitively."""
    target = _normalize_modality_lookup_token(token)
    if not target:
        return None
    for configured in configured_tokens:
        if _normalize_modality_lookup_token(configured) == target:
            return str(configured).strip()
    return None

# ⚠️  DEPRECATED: 此函数仅保留用于向后兼容双模态模型。
# 新代码应使用 resolve_multimodal_channel_contract() 和 N 模态槽位体系。
# 在 N 模态扩展后，此函数不再推荐使用。请迁移至新槽位体系 (A/B/C/...)。
def resolve_modality_pair(
    data_or_tokens: Optional[Any],
    *,
    default_secondary: Optional[str] = None,
    strict: bool = False,
) -> tuple[str, str]:
    """Resolve the primary and secondary modality tokens from configuration.

    precision contract 恢复的是值域/加载/增强/预处理语义，
    不是回退模态选择机制。
    resolve_modality_pair()/resolve_requested_modality_token() 继续作为唯一模态解析入口。

    Unlike ``resolve_rgb_x_pair``, this function does NOT require RGB to be
    present.  It simply treats ``modality_used[0]`` as primary and
    ``modality_used[1]`` as secondary.

    Returns:
        (primary_token, secondary_token)
    """
    data = data_or_tokens if isinstance(data_or_tokens, dict) else None
    configured_tokens = (
        list(data_or_tokens)
        if isinstance(data_or_tokens, Sequence) and not isinstance(data_or_tokens, (str, bytes, dict))
        else get_configured_modality_tokens(data)
    )
    configured_tokens = _coerce_configured_modalities(configured_tokens)

    if strict:
        if len(configured_tokens) < 2:
            raise ValueError(
                f"modality configuration must contain at least 2 modalities, got: {configured_tokens}"
            )
        # N≥3：primary=首个，secondary=其余拼接显示名（槽位体系才是通道真相，此处仅显示用途）
        if len(configured_tokens) == 2:
            return configured_tokens[0], configured_tokens[1]
        return configured_tokens[0], "+".join(configured_tokens[1:])

    if len(configured_tokens) >= 2:
        return configured_tokens[0], configured_tokens[1]

    # Single token: primary from config, secondary from fallback
    if configured_tokens:
        primary = configured_tokens[0]
    elif isinstance(data, dict):
        primary = _clean_modality_value(data.get("primary_modality")) or "rgb"
    else:
        primary = "rgb"

    # Fallback: try to extract secondary from dict keys
    if isinstance(data, dict):
        explicit_secondary = _clean_modality_value(data.get("x_modality"))
        if explicit_secondary and configured_tokens:
            explicit_secondary = find_configured_modality_token(explicit_secondary, configured_tokens) or explicit_secondary
        fallback_secondary = (
            explicit_secondary
            or (_clean_modality_value(default_secondary) if default_secondary is not None else None)
            or "unknown"
        )
        return primary, fallback_secondary

    # No dict: use default_secondary or "unknown"
    fallback = (_clean_modality_value(default_secondary) if default_secondary is not None else None) or "unknown"
    return primary, fallback


def resolve_rgb_x_pair(
    data_or_tokens: Optional[Any],
    *,
    default_x: Optional[str] = None,
    strict: bool = False,
) -> tuple[str, str]:
    """Backward-compatible wrapper around ``resolve_modality_pair``.

    Preserves the original function signature so that all existing callers
    continue to work without changes.
    """
    return resolve_modality_pair(
        data_or_tokens,
        default_secondary=default_x,
        strict=strict,
    )


def resolve_requested_modality_token(
    requested: Optional[str],
    data_or_tokens: Optional[Any],
    *,
    default_x: Optional[str] = None,
    strict: bool = False,
) -> Optional[str]:
    """Resolve user-facing modality tokens (RGB/X/concrete) to configured modality tokens."""
    if requested is None:
        return None

    configured_tokens = (
        list(data_or_tokens)
        if isinstance(data_or_tokens, Sequence) and not isinstance(data_or_tokens, (str, bytes, dict))
        else get_configured_modality_tokens(data_or_tokens if isinstance(data_or_tokens, dict) else None)
    )
    configured_tokens = _coerce_configured_modalities(configured_tokens)
    normalized = normalize_modality_token(requested)

    primary_token, secondary_token = resolve_modality_pair(
        configured_tokens if configured_tokens else data_or_tokens,
        default_secondary=default_x,
        strict=strict and bool(configured_tokens),
    )

    if normalized == "RGB":
        return primary_token
    if normalized == "X":
        return secondary_token

    resolved = find_configured_modality_token(normalized, configured_tokens) if configured_tokens else None
    if resolved:
        return resolved

    if strict:
        raise ValueError(f"requested modality {requested!r} not found in configured modalities: {configured_tokens}")

    cleaned = _clean_modality_value(normalized)
    return cleaned


def resolve_single_modal_build_spec(single_modal: str, data_dict: dict) -> SingleModalBuildSpec:
    """Resolve ``single_modal`` into a build-time routing spec.

    Args:
        single_modal: User-specified modality name (e.g. 'ir', 'depth', 'rgb', 'fuse').
        data_dict: Parsed dataset YAML dictionary.

    Returns:
        SingleModalBuildSpec with route token and metadata.

    Raises:
        ValueError: When the modality is not found in dataset modality_used.
    """
    if not isinstance(single_modal, str) or not single_modal.strip():
        raise ValueError("single_modal must be a non-empty string when provided")

    token = single_modal.strip()

    # 先按数据集 modality_used 匹配模态名（'depth'/'event' 等纯字母名会被
    # resolve_input_source_slot 误判为组合槽位 token，必须先于槽位解析）
    configured_tokens = get_configured_modality_tokens(data_dict)
    for idx, configured_token in enumerate(configured_tokens):
        if _normalize_modality_lookup_token(configured_token) == _normalize_modality_lookup_token(token):
            slot = chr(65 + idx) if idx < 26 else f"modality_{idx}"
            return SingleModalBuildSpec(
                raw=token,
                role_token=slot,
                role_name=slot,
                matched_modality=configured_token,
            )

    try:
        slot = resolve_input_source_slot(token)
    except ValueError:
        slot = None
    if slot is not None:
        return SingleModalBuildSpec(
            raw=token,
            role_token=slot,
            role_name=slot,
            matched_modality=None,
        )

    if not configured_tokens:
        raise ValueError(
            f"single_modal='{token}': dataset has no modality_used configured, "
            f"cannot resolve input role"
        )

    raise ValueError(
        f"single_modal='{token}' not found in dataset modality_used: {configured_tokens}"
    )



def resolve_single_modal_input_role(single_modal: str, data_dict: dict) -> str:
    """Resolve single-modal value to input role A/B/Fuse from dataset configuration."""
    return resolve_single_modal_build_spec(single_modal, data_dict).role_token


def has_multimodal_routing_layers(cfg_dict: dict) -> bool:
    """Check whether any layer in the model YAML config has a 5th-field routing annotation."""
    all_layers = cfg_dict.get("backbone", []) + cfg_dict.get("head", [])
    for layer_config in all_layers:
        if len(layer_config) >= 5:
            try:
                if resolve_input_source_slot(layer_config[4]) is not None:
                    return True
            except ValueError:
                continue
    return False


def has_multimodal_config(cfg_dict: dict) -> bool:
    """Check if model YAML config has multimodal routing configured.

    Checks both top-level ``input`` field and backbone/head layer 5th-field
    routing annotations.
    """
    if cfg_dict.get("input") is not None:
        try:
            if resolve_input_source_slot(cfg_dict["input"]) is not None:
                return True
        except ValueError:
            pass
    all_layers = cfg_dict.get("backbone", []) + cfg_dict.get("head", [])
    for layer_config in all_layers:
        if len(layer_config) >= 5:
            try:
                if resolve_input_source_slot(layer_config[4]) is not None:
                    return True
            except ValueError:
                continue
    return False


def _detect_slots_from_model_cfg(model_cfg: dict) -> list[str]:
    """从模型 YAML 第 5 字段扫描槽位字母（按字母序）。

    仅用于无 data 构建场景；每个槽位默认 3 通道（决策 §3.4）。
    第 5 字段无法解析为槽位 token 时直接报错。
    """
    detected: set[str] = set()
    for section in ("backbone", "head"):
        for layer_config in model_cfg.get(section, []) or []:
            if not isinstance(layer_config, (list, tuple)) or len(layer_config) < 5:
                continue
            token = layer_config[4]
            if token is None:
                continue
            try:
                resolved = resolve_input_source_slot(token)
            except ValueError:
                raise ValueError(
                    f"模型 YAML 第 5 字段 {token!r} 无法解析为槽位（backbone/head: {section}）"
                ) from None
            if resolved is None:
                continue
            if resolved == "Fuse" or str(token).casefold() == "mask":
                continue  # 伪源：Fuse 全槽拼接 / Mask availability 切片，不是槽位身份
            detected.update(resolved)
    return sorted(detected)


def resolve_multimodal_channel_contract(
    *,
    data: Optional[dict] = None,
    requested_modality: Optional[str] = None,
    model_cfg: Optional[Union[dict, str]] = None,
) -> MultiModalChannelContract:
    """唯一解析点：从 data.yaml（+模型 YAML 兜底）解析 N 模态通道契约（决策 §2.1/§3）。

    通道数解析顺序（每个模态独立）：
    1. ``modality_ch[name]`` 显式值
    2. 旧字段兼容：仅当 N == 2 时 ``Pch`` → 槽位 A、``Xch`` → 槽位 B；
       N ≥ 3 且出现 ``Xch``/``Pch`` → 报错，提示改写 ``modality_ch``
    3. 自动检测 ``auto_detect_modality_channels``（目录缺失/无图/不一致均报错）
    4. 显式值与检测值不一致 → 报错

    校验（fail-fast）：
    - ``modality_used`` 每个名字必须在 ``modality`` 中有目录
    - ``modality_ch`` 出现 ``modality_used`` 之外的键 → 报错（防拼写错误被吞）
    - 通道数必须是正整数
    - 无 data 时从模型 YAML 第 5 字段扫描槽位，每槽默认 3ch

    Args:
        data: 已解析的 data.yaml 字典
        requested_modality: 消融时指定的模态 token（仅用于 runtime token 解析）
        model_cfg: 模型 YAML 配置（dict 或 str 路径；str 会加载为 dict）

    Returns:
        MultiModalChannelContract
    """
    if isinstance(model_cfg, str):
        from ultralytics.nn.tasks import yaml_model_load

        model_cfg = yaml_model_load(model_cfg)
    dataset = data or {}

    modality_used = get_configured_modality_tokens(dataset)
    has_data_cfg = bool(modality_used)

    if not has_data_cfg:
        legacy_fields = [k for k in ("Pch", "Xch", "x_modality") if k in dataset]
        if legacy_fields or "modality_ch" in dataset:
            fields = legacy_fields or ["modality_ch"]
            raise ValueError(
                f"data.yaml 含多模态字段 {fields}，但未定义 modality_used/modality。"
                "请迁移为 N 模态契约，例如：\n"
                "  modality: {rgb: images, ir: images_ir}\n"
                "  modality_used: [rgb, ir]\n"
                "  Xch: 3\n"
                "N>=3 时请用 modality_ch 声明各槽位通道数。"
            )

    modality_ch_cfg = dataset.get("modality_ch") if isinstance(dataset.get("modality_ch"), dict) else None
    if modality_ch_cfg is not None:
        extra = set(modality_ch_cfg) - set(modality_used)
        if extra:
            raise ValueError(
                f"modality_ch 含 modality_used 之外的键 {sorted(extra)}，请检查拼写（modality_used={modality_used}）"
            )

    if has_data_cfg:
        modality_names = tuple(modality_used)
        modality_mapping = dataset.get("modality") or dataset.get("modalities") or {}
        missing = [n for n in modality_used if n not in modality_mapping]
        if modality_mapping and missing:
            raise ValueError(f"modality_used 中的名字 {missing} 未在 modality 目录映射中定义: {modality_mapping}")
        slots = tuple(resolve_modality_slots(modality_used))
        channels = []
        for name in modality_used:
            explicit = modality_ch_cfg.get(name) if modality_ch_cfg else None
            if explicit is not None:
                ch = int(explicit)
                if ch <= 0:
                    raise ValueError(f"modality_ch[{name!r}] = {explicit!r} 必须是正整数")
            elif len(modality_used) == 2 and ("Pch" in dataset or "Xch" in dataset):
                # 旧字段兼容，仅 N == 2（决策 §2.1.2）
                ch = int(dataset.get("Pch", 3)) if name == modality_used[0] else int(dataset.get("Xch", 3))
            else:
                if len(modality_used) >= 3 and ("Pch" in dataset or "Xch" in dataset):
                    raise ValueError(
                        f"N≥3 ({modality_used}) 不再支持 Pch/Xch，请改写 data.yaml 的 modality_ch 字段"
                    )
                ch = auto_detect_all_modality_channels(dataset, [name])[0]
            channels.append(ch)
        modality_channels = tuple(channels)
        is_mm = len(modality_used) >= 2
    else:
        # 无 data 构建（YOLOMM('x.yaml')）：从模型 YAML 第 5 字段扫描槽位，每槽默认 3ch（决策 §3.4）
        if isinstance(model_cfg, dict):
            slot_letters = _detect_slots_from_model_cfg(model_cfg)
        else:
            slot_letters = []
        if not slot_letters:
            slot_letters = ["A"]
        modality_names = tuple(f"modality_{chr(64 + i)}" for i in range(1, len(slot_letters) + 1))
        slots = tuple(slot_letters)
        modality_channels = tuple(3 for _ in slot_letters)
        is_mm = len(slot_letters) >= 2

    if isinstance(model_cfg, dict) and has_multimodal_routing_layers(model_cfg):
        is_mm = True

    mask_channels = 0
    if dataset.get("mm_use_mask"):
        if not has_data_cfg:
            raise ValueError("mm_use_mask 需要 data.yaml 定义 modality_used/modality")
        mask_channels = len(modality_used)
    elif not has_data_cfg and isinstance(model_cfg, dict):
        # 无 data 构建（YOLOMM('x.yaml')）：模型 YAML 第 5 字段含 'Mask' 源时按槽位数推导
        layers = []
        if isinstance(model_cfg.get("backbone"), list):
            layers.extend(model_cfg["backbone"])
        if isinstance(model_cfg.get("head"), list):
            layers.extend(model_cfg["head"])
        if any(len(lc) >= 5 and str(lc[4]).casefold() == "mask" for lc in layers if isinstance(lc, (list, tuple))):
            mask_channels = len(slots)

    return MultiModalChannelContract(
        modality_channels=modality_channels,
        modality_names=modality_names,
        slots=slots,
        is_multimodal_model=is_mm,
        mask_channels=mask_channels,
    )



def get_modality_path_from_data(data: Optional[dict], modality_name: str) -> str:
    """Return the configured modality directory using case-insensitive key matching.

    Resolution order:
    1. ``data["modality"]`` / ``data["modalities"]`` dictionary (exact match)
    2. Infer from ``modality_used`` + ``train`` path (primary) or ``images_xxx`` convention
    3. Hardcoded fallback: ``rgb -> "images"``, others -> ``"images_{name}"``
    """
    data = data or {}
    warn_deprecated_mm_fields(data)

    # Priority 1: explicit mapping in config
    mapping = data.get("modality") or data.get("modalities")
    if isinstance(mapping, dict):
        resolved_key = find_configured_modality_token(modality_name, mapping.keys())
        if resolved_key is not None and resolved_key in mapping:
            return mapping[resolved_key]

    # Priority 2: infer from modality_used + train path
    modality_used = data.get("modality_used") or data.get("models")
    if isinstance(modality_used, (list, tuple)) and modality_used:
        normalized = _normalize_modality_lookup_token(modality_name)
        first_normalized = _normalize_modality_lookup_token(modality_used[0])
        if normalized == first_normalized:
            # Primary modality: extract directory name from train path
            train_path = data.get("train", "").replace("\\", "/").strip("./")
            for segment in train_path.split("/"):
                if segment.startswith("images"):
                    return segment
            # Fallback: convention images_<modality> or "images" for rgb
            if normalized == "rgb":
                return "images"
            return f"images_{modality_name}"
        else:
            # Secondary modality: images_xxx convention
            return f"images_{modality_name}"

    # Priority 3: final fallback (original behaviour)
    if _normalize_modality_lookup_token(modality_name) == "rgb":
        return "images"
    return f"images_{modality_name}"


def normalize_modality_token(modality: Optional[str]) -> Optional[str]:
    """Normalize modality tokens for internal routing.

    Legacy: rgb -> RGB, x -> X
    New:    a -> PRIMARY, b -> SECONDARY, fuse -> FUSED
    Other concrete modality names (thermal/depth/ir/...) are kept as-is.
    """
    if modality is None:
        return None
    if not isinstance(modality, str):
        return modality
    m = modality.strip()
    low = m.lower()
    # Legacy aliases
    if low == "rgb":
        return "RGB"
    if low == "x":
        return "X"
    # New aliases -> resolve to internal roles
    if low == "a":
        return "PRIMARY"
    if low == "b":
        return "SECONDARY"
    if low == "fuse":
        return "FUSED"
    return m


def extract_x_modality_name_from_data(data: Optional[dict]) -> str:
    """训练期专用：从已解析的数据配置中提取 X 模态名字。"""
    data = data or {}

    explicit_x = _clean_modality_value(data.get("x_modality"))
    if explicit_x:
        configured_tokens = get_configured_modality_tokens(data)
        if configured_tokens:
            resolved = resolve_requested_modality_token(explicit_x, configured_tokens, default_x=explicit_x)
            if resolved and _normalize_modality_lookup_token(resolved) != "rgb":
                return resolved
        return explicit_x

    _rgb_token, x_token = resolve_rgb_x_pair(data, default_x="unknown")
    return x_token or "unknown"


def set_mm_contract_in_ckpt(ckpt: dict, contract: Any) -> dict:
    """把通道契约写入 ckpt（Wave 4.4：优先 mm_contract，旧 Xch 按 N==2 读取侧还原）。"""
    if ckpt is None or contract is None:
        return ckpt
    ckpt["mm_contract"] = contract.to_meta()
    metadata = ckpt.get(MM_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["mm_contract"] = dict(ckpt["mm_contract"])
    ckpt[MM_METADATA_KEY] = metadata
    return ckpt


def get_mm_contract_from_ckpt(ckpt: dict):
    """从 ckpt 还原契约；优先 mm_contract，旧 Xch 按 N==2 解释。"""
    if not isinstance(ckpt, dict):
        return None
    meta = ckpt.get("mm_contract")
    if meta is None:
        md = ckpt.get(MM_METADATA_KEY)
        if isinstance(md, dict):
            meta = md.get("mm_contract")
    if isinstance(meta, dict) and "modality_used" in meta:
        return MultiModalChannelContract.from_meta(meta)
    if "Xch" in ckpt:  # 旧 ckpt
        return MultiModalChannelContract.from_meta({"Xch": ckpt["Xch"], "Pch": ckpt.get("Pch", 3)})
    return None


def set_x_modality_name_in_ckpt_metadata(
    ckpt: dict,
    x_modality_name: Optional[str],
    primary_modality_name: Optional[str] = None,
) -> dict:
    """Write modality names to ckpt metadata."""
    metadata = ckpt.get(MM_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[MM_X_MODALITY_KEY] = str(x_modality_name or "unknown").strip() or "unknown"
    if primary_modality_name is not None:
        metadata["primary_modality_name"] = str(primary_modality_name).strip() or "rgb"
    ckpt[MM_METADATA_KEY] = metadata
    return ckpt


def validate_single_modal_and_modality_exclusive(args: Any) -> None:
    """Fail fast when build-time ``single_modal`` and runtime ``modality`` are mixed."""
    if args is None:
        return

    if isinstance(args, dict):
        single_modal = args.get("single_modal")
        modality = args.get("modality")
    else:
        single_modal = getattr(args, "single_modal", None)
        modality = getattr(args, "modality", None)

    if _clean_modality_value(single_modal) and _clean_modality_value(modality):
        raise ValueError(
            "single_modal and modality cannot be used together. "
            "single_modal is build-time YAML injection for plain single-modal models; "
            "modality is runtime ablation/filling for already-multimodal models."
        )


def get_single_modal_build_metadata_from_model(model: Any) -> Optional[dict[str, Any]]:
    """Return verified ``single_modal_build`` metadata from ``model.yaml`` if present."""
    for candidate in (model, getattr(model, "model", None)):
        yaml_dict = getattr(candidate, "yaml", None) if candidate is not None else None
        if isinstance(yaml_dict, dict):
            meta = yaml_dict.get(SINGLE_MODAL_BUILD_KEY)
            if isinstance(meta, dict) and meta and has_multimodal_config(yaml_dict):
                return dict(meta)
    return None


def set_single_modal_build_metadata_in_ckpt(ckpt: dict, model: Any = None, args: Any = None) -> dict:
    """Persist verified ``single_modal`` build metadata into checkpoint fields."""
    meta = get_single_modal_build_metadata_from_model(model)
    raw_arg = getattr(args, "single_modal", None) if args is not None else None
    if isinstance(args, dict):
        raw_arg = args.get("single_modal")

    if not meta:
        if _clean_modality_value(raw_arg):
            raise ValueError(
                f"single_modal={raw_arg!r} was requested, but model.yaml has no verified "
                f"{SINGLE_MODAL_BUILD_KEY!r} metadata. Refusing to mark checkpoint as multimodal."
            )
        return ckpt

    ckpt[SINGLE_MODAL_BUILD_KEY] = dict(meta)
    ckpt["is_multimodal"] = True
    metadata = ckpt.get(MM_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[SINGLE_MODAL_BUILD_KEY] = dict(meta)
    ckpt[MM_METADATA_KEY] = metadata
    return ckpt


def sync_x_modality_name_to_model(
    model: Any,
    x_modality_name: Optional[str],
    primary_modality_name: Optional[str] = None,
) -> str:
    """Sync modality names to model metadata, router, and module attributes."""
    secondary = str(x_modality_name or "unknown").strip() or "unknown"
    primary = str(primary_modality_name or "rgb").strip() or "rgb"

    for candidate in (model, getattr(model, "model", None)):
        if candidate is None:
            continue
        model_metadata = getattr(candidate, "metadata", None)
        if not isinstance(model_metadata, dict):
            model_metadata = {}
        model_metadata[MM_X_MODALITY_KEY] = secondary
        model_metadata["primary_modality_name"] = primary
        setattr(candidate, "metadata", model_metadata)

    # Sync to router
    router_chains = (
        ("mm_router",),
        ("multimodal_router",),
        ("model", "mm_router"),
        ("model", "multimodal_router"),
    )
    for chain in router_chains:
        obj = model
        for attr in chain:
            if not hasattr(obj, attr):
                obj = None
                break
            obj = getattr(obj, attr)
        if obj is not None:
            setattr(obj, "x_modality_type", secondary)
            if hasattr(obj, "primary_modality_type"):
                setattr(obj, "primary_modality_type", primary)

    # Sync to modules
    modules_fn = getattr(model, "modules", None)
    if callable(modules_fn):
        for module in modules_fn():
            if hasattr(module, "_mm_x_modality"):
                module._mm_x_modality = secondary
            if hasattr(module, "_mm_primary_modality"):
                module._mm_primary_modality = primary

    return secondary


def get_runtime_x_modality_name(model: Any) -> str:
    """Read X modality name from checkpoint metadata; fallback to 'unknown'."""
    for candidate in (model, getattr(model, "model", None)):
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict):
            name = metadata.get(MM_X_MODALITY_KEY)
            if isinstance(name, str) and name.strip():
                return name.strip()
    return "unknown"


def get_runtime_primary_modality_name(model: Any) -> str:
    """Read primary modality name from checkpoint metadata; fallback to 'rgb'."""
    for candidate in (model, getattr(model, "model", None)):
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict):
            name = metadata.get("primary_modality_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return "rgb"


def validate_mm_config_format(config):
    """Validate multimodal configuration format correctness（槽位语义）。

    第 5 字段无法解析为槽位 token 时直接报错（fail-fast）。
    """
    routing_layers: dict[str, list[int]] = {}

    for section in ['backbone', 'head']:
        for i, layer_config in enumerate(config.get(section, [])):
            if len(layer_config) == 5:
                slot = resolve_input_source_slot(layer_config[4])
                if slot is not None:
                    routing_layers.setdefault(slot, []).append(i)

    total = sum(len(v) for v in routing_layers.values())
    LOGGER.info(f"MultiModal: config validation complete, {total} routing layers")
    for slot, layers in sorted(routing_layers.items()):
        LOGGER.info(f"MultiModal: 槽位 {slot} routed={len(layers)}")

    return {
        **{f"{k.lower()}_layers": v for k, v in routing_layers.items()},
        'total_routing_layers': total,
    }


def mm_system_status():
    """Display multimodal system status"""
    LOGGER.info("MultiModal: dynamic modality routing system status check...")
    LOGGER.info("MultiModal: supports arbitrary dual-modality (Phase 1)")
    LOGGER.info("MultiModal: input aliases: RGB/X/Dual (legacy) + A/B/Fuse (new)")
    LOGGER.info("MultiModal: config format: [from, repeats, module, args, 'RGB'/'X'/'Dual'/'A'/'B'/'Fuse']")
    LOGGER.info("MultiModal: system version: v2.0 - universal modality routing")
    return True


def check_mm_model_attributes(model):
    """Check multimodal attributes in the model"""
    mm_layers = []
    
    for i, m in enumerate(model.model if hasattr(model, 'model') else []):
        if hasattr(m, '_mm_input_source'):
            layer_info = {
                'layer_index': getattr(m, '_mm_layer_index', i),
                'input_source': getattr(m, '_mm_input_source', None),
                'x_modality': getattr(m, '_mm_x_modality', 'unknown'),
                'version': getattr(m, '_mm_version', 'unknown'),
                'new_input_start': getattr(m, '_mm_new_input_start', False)
            }
            mm_layers.append(layer_info)
    
    if mm_layers:
        LOGGER.info(f"MultiModal: 发现 {len(mm_layers)} 个多模态路由层")
        for layer_info in mm_layers:
            source = layer_info['input_source']
            idx = layer_info['layer_index']
            if layer_info['new_input_start']:
                LOGGER.info(f"MultiModal: Layer {idx} - {source}模态新输入起点")
            else:
                LOGGER.info(f"MultiModal: Layer {idx} - {source}模态路由层")
    else:
        LOGGER.info("MultiModal: 未发现多模态路由层，使用标准模式")
    
    return mm_layers


def get_mm_system_info():
    """Get multimodal system information"""
    return {
        'version': 'v2.0',
        'supported_input_aliases': list(INPUT_SOURCE_ALIASES.keys()),
        'internal_roles': ['PRIMARY', 'SECONDARY', 'FUSED'],
        'supported_architectures': ['YOLO', 'RTDETR'],
    }


def check_tensor_dtype_for_half(tensor, context=""):
    """在 half() 转换前检查 float64 张量。"""
    import torch
    if tensor.dtype == torch.float64:
        raise ValueError(
            f"{context} Cannot convert float64 tensor to float16 — "
            f"this would lose twice the precision of a float32 cast. "
            f"Convert to float32 first."
        )


def prepare_multimodal_cfg_dict(cfg, data, contract=None):
    """Load model YAML as dict and inject ``mm_contract`` (+legacy ``dataset_config``).

    Shared utility called by all multimodal trainer ``get_model`` methods.
    契约只解析一次：``contract`` 由调用方 resolve 后传入，Router 直接复用（决策 §3.2）。

    Args:
        cfg: Model config (str path or dict).
        data: Dataset config dict。
        contract: 已解析的 MultiModalChannelContract（推荐）。

    Returns:
        dict: cfg_dict ready for DetectionModel / SegmentationModel / etc.
        Falls back to original *cfg* when YAML loading fails.
    """
    from copy import deepcopy

    if isinstance(cfg, str):
        try:
            from ultralytics.nn.tasks import yaml_model_load
            cfg_dict = yaml_model_load(cfg)
        except Exception:
            return cfg
    elif isinstance(cfg, dict):
        cfg_dict = deepcopy(cfg)
    else:
        return cfg

    if contract is not None:
        cfg_dict['mm_contract'] = contract
    cfg_dict['dataset_config'] = dict(data) if data else {}
    return cfg_dict


def inject_single_modal_yaml(cfg_dict: dict, single_modal: str, data_dict: dict) -> dict:
    """Build-time YAML injection: convert a plain single-modal YAML to multimodal format.

    1. Resolve ``single_modal`` + ``data_dict`` -> role (A/B/Fuse).
    2. Inject ``input: <role>`` at YAML top level.
    3. Annotate the first backbone layer with the 5th-field routing marker.
    4. Return the modified cfg_dict (in-place, also returned for convenience).

    This is the ONLY place where single_modal drives YAML mutation.
    All downstream code (router, trainer, model) consumes the already-injected YAML.

    Raises:
        ValueError: If single_modal cannot be resolved from data_dict.
    """
    spec = resolve_single_modal_build_spec(single_modal, data_dict)

    # Top-level input marker (persistent for .pt saves)
    cfg_dict["input"] = spec.role_token
    cfg_dict[SINGLE_MODAL_BUILD_KEY] = spec.to_metadata()

    # Annotate first backbone layer
    backbone = cfg_dict.get("backbone", [])
    if not backbone:
        raise ValueError("single_modal YAML injection requires a non-empty backbone")
    first = list(backbone[0])
    if len(first) == 4:
        first.append(spec.role_token)
        backbone[0] = first
    elif len(first) >= 5:
        first[4] = spec.role_token
        backbone[0] = first
    else:
        raise ValueError(f"Invalid first backbone layer for single_modal injection: {first}")

    return cfg_dict


def prepare_multimodal_build_cfg(cfg, data, args: Any = None, logger: Any = None, contract=None):
    """Prepare a model cfg with build-time ``single_modal`` injection and dataset config.

    ``single_modal`` routing must be injected before model parsing so every task
    consumes the same already-mutated YAML shape.  This helper centralizes that
    build-time step for YOLOMM/RTDETRMM trainers.
    """
    from copy import deepcopy

    log = logger or LOGGER
    validate_single_modal_and_modality_exclusive(args)
    cfg_in = cfg or (getattr(args, "model", None) if args is not None else None)
    single_modal = getattr(args, "single_modal", None) if args is not None else None

    if not single_modal:
        return prepare_multimodal_cfg_dict(cfg_in, data, contract=contract)

    cfg_dict_raw = None
    if isinstance(cfg_in, str):
        from ultralytics.nn.tasks import yaml_model_load

        cfg_dict_raw = yaml_model_load(cfg_in)
    elif isinstance(cfg_in, dict):
        cfg_dict_raw = deepcopy(cfg_in)
    else:
        raise ValueError(f"single_modal={single_modal!r} requires a YAML/dict model config, got {type(cfg_in).__name__}")

    if isinstance(cfg_dict_raw, dict):
        if has_multimodal_config(cfg_dict_raw):
            raise ValueError(
                f"single_modal='{single_modal}' can only be used with a plain single-modal YAML. "
                "The provided model YAML already has multimodal routing."
            )
        data_dict = dict(data) if isinstance(data, dict) else {}
        data_path = getattr(args, "data", None) if args is not None else None
        if data_path:
            from ultralytics.utils import YAML, checks

            data_dict = YAML.load(checks.check_yaml(data_path))
        if not data_dict:
            raise ValueError(f"single_modal='{single_modal}' requires dataset config to resolve A/B/Fuse role")
        inject_single_modal_yaml(cfg_dict_raw, single_modal, data_dict)
        cfg_in = cfg_dict_raw
        if log:
            log.info(
                f"single_modal='{single_modal}' injected as input={cfg_dict_raw.get('input')} "
                "for build-time multimodal routing."
            )

    return prepare_multimodal_cfg_dict(cfg_in, data, contract=contract)
