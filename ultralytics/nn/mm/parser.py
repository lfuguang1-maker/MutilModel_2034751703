# Ultralytics Multimodal Config Parser
# Universal YAML configuration parsing for N-modality architectures
# Version: v3.0

from ultralytics.utils import LOGGER
from .utils import resolve_input_source_slot


def _slot_resolves(token) -> bool:
    """token 能否解析为槽位身份（含组合槽与 Fuse）。"""
    try:
        return resolve_input_source_slot(token) is not None
    except ValueError:
        return False


class MultiModalConfigParser:
    """
    Universal Multimodal Configuration Parser

    Handles YAML configuration parsing for both YOLO and RTDETR
    with dynamic modality routing (5-field layer config).
    槽位身份：A/B/C/...（≤26）+ Fuse；别名 RGB/X/Dual 解析后立刻落槽位。
    """

    def __init__(self):
        # 槽位字母 + Fuse + legacy 别名
        self.supported_input_sources = [chr(65 + i) for i in range(26)] + ["Fuse", "RGB", "X", "Dual"]

    def validate_config_format(self, config):
        """Validate multimodal configuration format correctness（槽位语义）"""
        slot_layers: dict[str, list[int]] = {}

        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) == 5:
                    slot = resolve_input_source_slot(layer_config[4])
                    if slot is not None:
                        slot_layers.setdefault(slot, []).append(i)

        total = sum(len(v) for v in slot_layers.values())
        LOGGER.info(f"MultiModal: config validation complete, {total} routing layers")
        return {f'{k.lower()}_layers': v for k, v in slot_layers.items()} | {'total_routing_layers': total}

    def extract_multimodal_info(self, config):
        """Extract multimodal information from configuration"""
        x_modality_type = config.get('dataset_config', {}).get('x_modality', 'unknown')
        primary_modality_type = config.get('dataset_config', {}).get('primary_modality', 'rgb')

        mm_layer_count = 0
        for section in ['backbone', 'head']:
            for layer_config in config.get(section, []):
                if len(layer_config) >= 5 and _slot_resolves(layer_config[4]):
                    mm_layer_count += 1

        return {
            'x_modality_type': x_modality_type,
            'primary_modality_type': primary_modality_type,
            'mm_layer_count': mm_layer_count,
            'supports_multimodal': mm_layer_count > 0,
        }

    def parse_config(self, config: dict) -> dict:
        """
        Build a minimal multimodal model_config dict for MultiModalRouter.

        Detects whether YAML has any 5th-field multimodal routing markers.
        """
        has_mm = False
        input_layers = []
        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) >= 5 and _slot_resolves(layer_config[4]):
                    has_mm = True
                    input_layers.append((section, i))
        out = dict(config)
        out['has_multimodal_layers'] = has_mm
        out['input_layers'] = input_layers
        return out
