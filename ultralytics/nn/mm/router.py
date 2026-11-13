# Ultralytics Multimodal Router - Universal RGB+X Data Routing
# Supports YOLO and RTDETR with zero-copy tensor routing
# Version: v1.0

from typing import Optional

import torch
import torch.nn as nn
from ultralytics.utils import LOGGER
from .filling import generate_modality_filling, adapt_xch
from .utils import extract_x_modality_name_from_data, is_rgb_modality, resolve_modality_pair


class MultiModalRouter:
    """
    Universal RGB+X Multimodal Data Router
    
    Supports:
    - RGB: 3-channel visible light images
    - X: 3-channel unified other modality (depth/thermal/lidar/etc.)
    - Dual: 6-channel RGB+X concatenated input
    
    Features:
    - Zero-copy tensor view routing
    - Configuration-driven data flow
    - Support for both YOLO and RTDETR architectures
    """
    
    # Backward-compat defaults (for unpickled legacy objects)
    runtime_modality = None
    runtime_strategy = None
    runtime_seed = None

    def __init__(self, config_dict=None, verbose=True):
        """Initialize multimodal router from the channel contract (N-modal).

        契约来源（决策 §3）：
        1. ``config_dict['mm_contract']``：已解析的 MultiModalChannelContract（唯一推荐路径）
        2. 无 data 构建：从模型 YAML 第 5 字段扫描槽位，每槽默认 3ch
        Router 不再自行 resolve 契约。
        """
        from .utils import resolve_multimodal_channel_contract, _detect_slots_from_model_cfg

        self.x_modality_type = 'unknown'
        self.primary_modality_type = 'rgb'
        self.verbose = verbose

        contract = None
        if config_dict and isinstance(config_dict.get('mm_contract'), object) and hasattr(config_dict.get('mm_contract'), 'slots'):
            contract = config_dict['mm_contract']
        elif config_dict and 'dataset_config' in config_dict:
            # 兼容：dataset_config 仍在时解析一次契约（不应发生；get_model 必须传 mm_contract）
            contract = resolve_multimodal_channel_contract(data=config_dict['dataset_config'])
        else:
            slot_letters = _detect_slots_from_model_cfg(config_dict) if isinstance(config_dict, dict) else []
            if not slot_letters:
                slot_letters = ['A']
            names = tuple(f'modality_{chr(64 + i)}' for i in range(1, len(slot_letters) + 1))
            # 模型 YAML 第 5 字段含 'Mask' 源时按槽位数推导 mask_channels（YOLOMM('x.yaml') 路径）
            has_mask_src = False
            if isinstance(config_dict, dict):
                for key in ('backbone', 'head'):
                    for lc in (config_dict.get(key) or []):
                        if isinstance(lc, (list, tuple)) and len(lc) >= 5 and str(lc[4]).casefold() == 'mask':
                            has_mask_src = True
                            break
                    if has_mask_src:
                        break
            from .utils import MultiModalChannelContract
            contract = MultiModalChannelContract(
                modality_channels=tuple(3 for _ in slot_letters),
                modality_names=names,
                slots=tuple(slot_letters),
                is_multimodal_model=len(slot_letters) >= 2,
                mask_channels=len(slot_letters) if has_mask_src else 0,
            )

        self.contract = contract
        self.modality_channels = contract.modality_channels
        self.modality_names = contract.modality_names
        self.slots = contract.slots
        self._slices = contract.slices
        # 兼容属性（只读语义，构网后不得修改）
        self.primary_ch = contract.primary_ch
        self.secondary_ch = contract.secondary_ch

        # INPUT_SOURCES 只含槽位与 Fuse（决策 §7 Wave 2.2）
        self.INPUT_SOURCES = {slot: contract.channels_of(slot) for slot in self.slots}
        self.INPUT_SOURCES['Fuse'] = contract.fused_ch
        self.mask_channels = contract.mask_channels
        if contract.mask_channels > 0:
            self.INPUT_SOURCES['Mask'] = contract.mask_channels
        if len(self.modality_names) >= 1:
            self.primary_modality_type = self.modality_names[0]
        if len(self.modality_names) >= 2:
            self.x_modality_type = self.modality_names[1]

        # 记录构建时通道契约（不随 update_dataset_config 改变）
        self._build_contract = {
            'modality_channels': self.modality_channels,
            'modality_names': self.modality_names,
            'slots': self.slots,
            'primary_ch': self.primary_ch,
            'secondary_ch': self.secondary_ch,
            'fused_ch': contract.fused_ch,
        }

        self.has_multimodal_config = self._detect_multimodal_config(config_dict)
        self.original_spatial_size = None
        self.original_inputs = {}
        self.runtime_modality = None
        self.runtime_strategy = None
        self.runtime_seed = None

        if self.verbose:
            LOGGER.info("MultiModal: router initialized (N-modal, contract-driven)")
            modal_info = ' + '.join(f"{name}({slot},{ch}ch)" for slot, name, ch in zip(self.slots, self.modality_names, self.modality_channels))
            LOGGER.info(f"MultiModal: {modal_info}, fused={contract.fused_ch}ch")

    def _detect_slots_from_yaml(self, config_dict):
        """从 YAML 第 5 字段扫描槽位字母（构网阶段无契约时使用）。"""
        if not config_dict or 'backbone' not in config_dict:
            return ['A']
        detected = set()
        for layer_cfg in config_dict['backbone']:
            if not isinstance(layer_cfg, list) or len(layer_cfg) < 5:
                continue
            token = layer_cfg[4]
            if not token or not isinstance(token, str):
                continue
            from .utils import resolve_input_source_slot
            try:
                resolved = resolve_input_source_slot(token)
            except ValueError:
                continue
            if resolved and resolved != 'Fuse':
                detected.update(resolved)
        return sorted(detected) if detected else ['A']

    def set_runtime_params(self, modality: str | None, strategy: str | None = None, seed: int | None = None):
        """Set runtime ablation/filling params used during setup_multimodal_routing()."""
        self.runtime_modality = (modality.lower() if isinstance(modality, str) else None)
        self.runtime_strategy = strategy
        self.runtime_seed = seed

    # --- Backward compatibility helpers for unpickled legacy routers ---
    def __setstate__(self, state):
        """Restore state with legacy checkpoint migration."""
        self.__dict__.update(state)
        # Runtime defaults
        for attr in ('runtime_modality', 'runtime_strategy', 'runtime_seed'):
            if attr not in self.__dict__:
                setattr(self, attr, None)
        self.original_spatial_size = None

        # --- Legacy checkpoint migration (in-memory only) ---
        if hasattr(self, 'INPUT_SOURCES') and isinstance(self.INPUT_SOURCES, dict):
            old_keys = {'RGB', 'X', 'Dual'}
            if old_keys & set(self.INPUT_SOURCES.keys()):
                # Migrate: RGB->PRIMARY, X->SECONDARY, Dual->FUSED
                self.primary_ch = self.INPUT_SOURCES.pop('RGB', 3)
                self.secondary_ch = self.INPUT_SOURCES.pop('X', 3)
                self.INPUT_SOURCES.pop('Dual', None)
                self.INPUT_SOURCES['PRIMARY'] = self.primary_ch
                self.INPUT_SOURCES['SECONDARY'] = self.secondary_ch
                self.INPUT_SOURCES['FUSED'] = self.primary_ch + self.secondary_ch

        # Migrate original_inputs keys
        old_inputs = getattr(self, 'original_inputs', {})
        new_inputs = {}
        for old_key in ('RGB', 'X', 'Dual'):
            if old_key in old_inputs:
                new_key = INPUT_SOURCE_ALIASES.get(old_key, old_key)
                new_inputs[new_key] = old_inputs[old_key]
        if new_inputs:
            self.original_inputs = new_inputs
        else:
            self.original_inputs = {}

        # Ensure primary_modality_type exists
        if not hasattr(self, 'primary_modality_type'):
            self.primary_modality_type = 'rgb'
        if not hasattr(self, 'primary_ch'):
            self.primary_ch = self.INPUT_SOURCES.get('PRIMARY', 3)
        if not hasattr(self, 'secondary_ch'):
            self.secondary_ch = self.INPUT_SOURCES.get('SECONDARY', 3)

        # 补齐 _build_contract（旧模型无此字段）
        if not hasattr(self, '_build_contract'):
            self._build_contract = {
                'primary_ch': self.primary_ch,
                'secondary_ch': self.secondary_ch,
                'fused_ch': self.primary_ch + self.secondary_ch,
            }

    def __getstate__(self):
        """
        自定义序列化：剔除运行时缓存，避免把数据 batch 张量写入权重文件。

        说明：
        - MultiModalRouter 在每次 forward 中会缓存上一批次输入（original_inputs），用于空间重置等运行时逻辑；
          这些张量与模型权重无关，且体积巨大，序列化会导致 .pt 异常虚高并携带数据内容。
        - 这里仅移除运行时缓存，不影响模型参数与结构的保存/恢复。
        """
        state = dict(self.__dict__)
        state['original_spatial_size'] = None
        state['original_inputs'] = {'PRIMARY': None, 'SECONDARY': None, 'FUSED': None}
        return state

    def _ensure_runtime_defaults(self):
        """Safeguard: define runtime fields if missing on legacy instances."""
        if not hasattr(self, 'runtime_modality'):
            self.runtime_modality = None
        if not hasattr(self, 'runtime_strategy'):
            self.runtime_strategy = None
        if not hasattr(self, 'runtime_seed'):
            self.runtime_seed = None
    
    def parse_layer_config(self, layer_config, layer_index, ch, verbose=True):
        """
        Parse layer configuration with optional 5th field for multimodal routing

        Args:
            layer_config: Layer configuration [from, repeats, module, args, input_source?]
            layer_index: Current layer index
            ch: Channel information
            verbose: Whether to print verbose information

        Returns:
            tuple: (input_channels, mm_input_source, mm_attributes)

        Raises:
            ValueError: When layer_config contains more than 5 fields (6th-field HOOK has been removed).
        """
        # Hard reject: 6th field (HOOK) is no longer supported
        if len(layer_config) > 5:
            raise ValueError(
                f"Layer {layer_index}: layer_config has {len(layer_config)} fields, but only 5 are supported "
                f"(from, repeats, module, args, input_source). "
                f"The 6th-field HOOK system has been removed. "
                f"Please update your model YAML to remove the 6th field."
            )

        # Parse standard 4 fields and optional 5th field (MM input source identifier)
        if len(layer_config) >= 5:
            f, n, m, args, mm_input_source = layer_config[:5]
        else:
            f, n, m, args = layer_config[:4]
            mm_input_source = None

        mm_attributes = {}
        c1 = None

        # Check 5th field: MM input source routing processing（槽位语义，fail-fast）
        if mm_input_source:
            from .utils import resolve_input_source_slot
            # Mask 伪源：availability mask 切片，不是槽位身份，不参与新起点逻辑（构建方案 §4.7）
            if isinstance(mm_input_source, str) and mm_input_source.casefold() == 'mask':
                if self.contract.mask_channels <= 0:
                    raise ValueError(
                        f"Layer {layer_index}: 使用 'Mask' 输入源但契约 mask_channels=0；"
                        "请在 data.yaml 设置 mm_use_mask: true"
                    )
                c1 = self.contract.mask_channels
                mm_attributes = {
                    '_mm_input_source': 'Mask',
                    '_mm_layer_index': layer_index,
                    '_mm_version': 'v1.0',
                    '_mm_x_modality': self.x_modality_type,
                    '_mm_primary_modality': self.primary_modality_type,
                }
                if verbose:
                    m_name = m.__name__ if hasattr(m, '__name__') else m
                    LOGGER.info(f"MultiModal Layer {layer_index}: '{m_name}' <- Mask 伪源 ({c1}通道)")
                return c1, mm_input_source, mm_attributes
            # 解析失败或不在契约槽位中 → 直接报错，不回退（决策 §6）
            slot = resolve_input_source_slot(mm_input_source, slots=self.slots)
            c1 = self.contract.channels_of(slot)

            m_name = m.__name__ if hasattr(m, '__name__') else m
            if slot == 'Fuse':
                modal_desc = "Fuse(全槽位拼接)"
            elif len(slot) == 1:
                slot_idx = self.slots.index(slot)
                modal_desc = f"槽位{slot}({self.modality_names[slot_idx]})"
            else:
                modal_desc = f"组合槽{slot}"

            mm_attributes = {
                '_mm_input_source': slot,
                '_mm_layer_index': layer_index,
                '_mm_version': 'v1.0',
                '_mm_x_modality': self.x_modality_type,
                '_mm_primary_modality': self.primary_modality_type,
            }

            # 非 A 槽位且 from=-1：一律标记新输入起点并记录槽位（决策 §7 Wave 2.3）
            if slot != 'A' and slot != 'Fuse' and f == -1:
                mm_attributes['_mm_new_input_start'] = True
                mm_attributes['_mm_new_start_slot'] = slot if len(slot) == 1 else None
                mm_attributes['_mm_spatial_reset'] = True
                if verbose:
                    LOGGER.info(f"MultiModal Layer {layer_index}: {modal_desc}模态新输入起点 (from=-1被重定向到{slot})")
                    LOGGER.info(f"MultiModal Layer {layer_index}: 空间重置标记已设置 (尺寸将从输入动态获取)")

            if verbose:
                LOGGER.info(f"MultiModal Layer {layer_index}: '{m_name}' <- {modal_desc}模态输入 ({c1}通道)")

        # 无第 5 字段：标准 c1 推导
        if c1 is None:
            # Handle both single index and list of indices
            if isinstance(f, list):
                if len(f) == 1:
                    f_idx = f[0]
                    c1 = ch[f_idx] if f_idx != -1 else ch[-1]
                else:
                    # Multiple inputs case, calculate total channels
                    c1 = sum(ch[i] if i != -1 else ch[-1] for i in f)
            else:
                c1 = ch[f] if f != -1 else ch[-1]

        return c1, mm_input_source, mm_attributes
    
    def setup_multimodal_routing(self, x, profile=False):
        """Setup multimodal input sources (N-modal, contract-driven, fail-fast).

        通道数对不上就报错，不许关路由或回退默认值（决策 §1/§7 Wave 2.6）：
        - ``got == fused_ch``：按契约 slices 切分各槽位
        - ``got == channels[0]``：单槽消融，其余槽位按各自通道数填充，Fuse 由全部槽位拼接
        - 其它通道数：raise（带 got / expected / slots / names / channels）

        Args:
            x: Input tensor
            profile: Whether to print profiling information

        Returns:
            tuple: (routing_enabled, input_sources_dict)
        """
        from .filling import generate_modality_filling

        # Backward-compat: make sure runtime_* fields exist even for unpickled legacy objects
        self._ensure_runtime_defaults()

        got = int(x.shape[1])
        fused_ch = self.contract.fused_ch
        mask_ch = self.contract.mask_channels

        # input_ch 布局 [data(fused_ch) | mask(M)]：Mask 伪源零拷贝切片
        if mask_ch > 0 and got == fused_ch + mask_ch:
            input_sources = {slot: x[:, sl, :, :] for slot, sl in zip(self.slots, self._slices)}
            input_sources['Fuse'] = x[:, :fused_ch, :, :]
            input_sources['Mask'] = x[:, fused_ch:, :, :]
            self.original_spatial_size = (x.shape[2], x.shape[3])
            self.cache_original_inputs(input_sources)
            return True, input_sources

        if got == fused_ch:
            input_sources = {slot: x[:, sl, :, :] for slot, sl in zip(self.slots, self._slices)}
            input_sources['Fuse'] = x
            if mask_ch > 0:
                # 无 mask 输入：默认全可用
                input_sources['Mask'] = torch.ones(x.shape[0], mask_ch, x.shape[2], x.shape[3],
                                                   dtype=x.dtype, device=x.device)
            self.original_spatial_size = (x.shape[2], x.shape[3])
            self.cache_original_inputs(input_sources)
            return True, input_sources

        if got == self.modality_channels[0] and len(self.slots) > 1:
            # 单槽消融：A 路输入，其余槽位按各自通道数生成填充
            self.original_spatial_size = (x.shape[2], x.shape[3])
            input_sources = {'A': x}
            for i, slot in enumerate(self.slots[1:], start=1):
                input_sources[slot] = generate_modality_filling(
                    x,
                    self.modality_names[0],
                    self.modality_names[i],
                    target_ch=self.modality_channels[i],
                    strategy=self.runtime_strategy,
                )
            input_sources['Fuse'] = torch.cat([input_sources[s] for s in self.slots], dim=1)
            if mask_ch > 0:
                # 单 A 模态：A 可用，其余 mask=0
                mask = torch.zeros(x.shape[0], mask_ch, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)
                mask[:, 0] = 1.0
                input_sources['Mask'] = mask
            self.cache_original_inputs(input_sources)
            return True, input_sources

        raise RuntimeError(
            f"MultiModal: 输入通道数 {got} 与契约不符（契约: slots={list(self.slots)}, "
            f"names={list(self.modality_names)}, channels={list(self.modality_channels)}, "
            f"fused_ch={fused_ch}, mask_ch={mask_ch}）。多模态模型不接受融合通道数以外的输入；"
            f"如需单模态推理请提供槽位 A 的 {self.modality_channels[0]} 通道输入。"
        )


        
    def route_layer_input(self, x, module, input_sources, profile=False):
        """
        Route input data for a specific layer based on its MM attributes
        
        Args:
            x: Current input tensor
            module: Current module
            input_sources: Multimodal input sources dictionary
            profile: Whether to print profiling information
            
        Returns:
            torch.Tensor or None: Routed input tensor, None if no routing needed
        """
        if not hasattr(module, '_mm_input_source'):
            return None
            
        # Validate input sources availability
        if not input_sources:
            if profile:
                LOGGER.warning(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - 输入源不可用")
            return None
            
        mm_input_source = module._mm_input_source

        # Mask 伪源：直接返回 mask 切片，不做槽位校验/拼接/新起点逻辑
        if mm_input_source == 'Mask':
            if 'Mask' not in input_sources:
                raise RuntimeError(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                    f"Mask 输入源不可用（契约 mask_channels={self.contract.mask_channels}）"
                )
            return input_sources['Mask']
        # 组合槽（AB/BC/...）：运行期按字母拼接槽位张量
        if isinstance(mm_input_source, str) and mm_input_source not in input_sources and len(mm_input_source) > 1 and mm_input_source != 'Fuse':
            slot_chars = [c for c in mm_input_source if c in self.slots]
            if slot_chars and len(slot_chars) == len(mm_input_source):
                input_sources[mm_input_source] = torch.cat([input_sources[s] for s in mm_input_source], dim=1)

        # ===== 路由逻辑：非 A 槽位新输入起点 =====
        if getattr(module, '_mm_new_input_start', False):
            # 新起点槽位必须显式记录，缺失即报错，不默认 'B'（决策 §7 Wave 2.5）
            new_start_slot = getattr(module, '_mm_new_start_slot', None)
            if not new_start_slot:
                raise RuntimeError(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 标记了 _mm_new_input_start "
                    f"但没有 _mm_new_start_slot；组合槽新起点请在路由前先拼接或拆分为单槽。"
                )
            if new_start_slot not in input_sources:
                raise RuntimeError(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                    f"{new_start_slot}槽位新输入起点需要{new_start_slot}输入源（可用: {list(input_sources)}）"
                )

            routed_x = input_sources[new_start_slot]

            # 验证新起点槽位数据通道数正确
            expected_channels = self.contract.channels_of(new_start_slot)
            if routed_x.shape[1] != expected_channels:
                raise RuntimeError(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                    f"{new_start_slot}槽位新输入起点期望{expected_channels}通道，但接收到{routed_x.shape[1]}通道"
                )

            if profile:
                slot_index = self.slots.index(new_start_slot)
                modal_name = self.modality_names[slot_index] if slot_index < len(self.modality_names) else 'unknown'
                LOGGER.info(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - {new_start_slot}({modal_name})槽位新输入起点"
                )
                LOGGER.info(f"MultiModal: 输入切换 {x.shape} -> {routed_x.shape}")
        else:
            # Normal modality routing - validate and use requested modality
            if mm_input_source not in input_sources:
                raise RuntimeError(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                    f"请求的输入源 '{mm_input_source}' 不存在（可用: {list(input_sources)}）"
                )

            routed_x = input_sources[mm_input_source]

            if profile:
                LOGGER.info(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 路由到 '{mm_input_source}' - 输入形状: {x.shape} -> {routed_x.shape}"
                )

        return routed_x
    
    def set_module_attributes(self, module, mm_attributes):
        """Set multimodal attributes on a module"""
        for attr_name, attr_value in mm_attributes.items():
            setattr(module, attr_name, attr_value)
            
    def get_original_spatial_size(self):
        """Get the original input spatial size for spatial reset"""
        return self.original_spatial_size
    
    def cache_original_inputs(self, input_sources):
        """Cache original per-slot inputs (zero-copy refs) for spatial reset."""
        self.original_inputs = {slot: input_sources.get(slot) for slot in list(self.slots) + ["Fuse", "Mask"]}

    def get_original_x_input(self, target_size=None, slot: Optional[str] = None):
        """Get original input of a slot (default B) with optional target size."""
        key = slot if slot is not None else (self.slots[1] if len(self.slots) > 1 else None)
        if key is None:
            return None
        x_input = self.original_inputs.get(key)
        if x_input is None:
            return None
        if target_size and target_size != x_input.shape[2:4]:
            pass  # 假设原始输入已是目标尺寸
        return x_input

    def reset_spatial_input(self, x, module, mm_input_sources, profile=False):
        """Reset a new-input-start slot to its original spatial size.

        重置目标从 ``_mm_new_start_slot`` 读取；缺失即报错（决策 §7 Wave 2.5）。
        """
        if not getattr(module, '_mm_new_input_start', False):
            return x

        slot = getattr(module, '_mm_new_start_slot', None)
        if not slot:
            raise RuntimeError(
                f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 缺少 _mm_new_start_slot"
            )
        if not mm_input_sources or slot not in mm_input_sources:
            raise RuntimeError(
                f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 缺少{slot}槽位输入源"
            )
        if self.original_spatial_size is None:
            raise RuntimeError(
                f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 无法获取原始尺寸"
            )

        reset_x = mm_input_sources[slot]
        if profile:
            LOGGER.info(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置完成（槽位 {slot}）")
            LOGGER.info(f"MultiModal: 尺寸重置 {x.shape} -> {reset_x.shape}")
        return reset_x

    def assert_contract_matches(self, contract):
        """构网后只断言：传入契约与本 router 的构建契约不一致即报错（决策 §7 Wave 2.7）。"""
        mismatches = []
        if tuple(getattr(contract, 'slots', ())) != tuple(self.slots):
            mismatches.append(f"slots {getattr(contract, 'slots', None)} != {self.slots}")
        if tuple(getattr(contract, 'modality_channels', ())) != tuple(self.modality_channels):
            mismatches.append(
                f"channels {getattr(contract, 'modality_channels', None)} != {self.modality_channels}"
            )
        if mismatches:
            raise RuntimeError(
                "MultiModal: 构建后通道契约不一致: " + "; ".join(mismatches)
                + f"（router 构建契约: slots={list(self.slots)}, names={list(self.modality_names)}, "
                f"channels={list(self.modality_channels)}）"
            )

    def update_dataset_config(self, dataset_config, update_x_channels: bool = True, update_x_modality: bool = True):
        """更新显示名；通道部分已删除，改为 ``assert_contract_matches``（决策 §7 Wave 2.7）。

        Args:
            dataset_config: Dataset configuration dict。
            update_x_channels: 忽略（保留签名兼容，通道不再可写）。
            update_x_modality: 是否刷新模态显示名。
        """
        if not dataset_config:
            return
        if not update_x_modality:
            return

        x_mod = extract_x_modality_name_from_data(dataset_config if isinstance(dataset_config, dict) else None)
        if x_mod and x_mod != 'unknown':
            self.x_modality_type = str(x_mod)

        if 'primary_modality' in dataset_config:
            primary_mod = dataset_config['primary_modality']
            if primary_mod and isinstance(primary_mod, str) and primary_mod.strip():
                self.primary_modality_type = primary_mod.strip()

    def _detect_multimodal_config(self, config_dict):
        """Detect if the configuration contains multimodal layers (slot tokens)."""
        from .utils import resolve_input_source_slot

        if not config_dict:
            return False

        if config_dict.get("input") is not None:
            try:
                if resolve_input_source_slot(config_dict["input"]) is not None:
                    return True
            except ValueError:
                pass

        all_layers = config_dict.get('backbone', []) + config_dict.get('head', [])
        for layer_config in all_layers:
            if len(layer_config) >= 5:
                try:
                    if resolve_input_source_slot(layer_config[4]) is not None:
                        return True
                except ValueError:
                    continue
        return False
