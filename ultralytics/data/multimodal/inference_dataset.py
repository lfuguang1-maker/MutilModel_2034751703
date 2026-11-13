# Ultralytics Multimodal Inference - Inference Dataset
# Dedicated dataset for multimodal inference (images only)
# Version: v1.0
# Date: 2026-01-13

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER
from ultralytics.data.augment import LetterBox
from ultralytics.nn.mm.utils import is_rgb_modality
from .precision_contract import resolve_precision_contract
from .utils import align_and_validate_x, letterbox_with_ratio_pad, to_tensor_rgb, to_tensor_x


def _load_primary_image(path) -> np.ndarray:
    """加载主模态图像，支持标准图像/NPY/NPZ/TIFF格式。"""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == '.npy':
        img = np.load(str(p))
        if img is None:
            raise ValueError(f"无法加载 .npy 文件: {p}")
        return img
    elif suffix == '.npz':
        with np.load(str(p)) as npz_file:
            preferred_keys = ('image', 'arr_0', 'array', 'data')
            selected_key = next((k for k in preferred_keys if k in npz_file.files), None)
            if selected_key is None:
                if len(npz_file.files) == 1:
                    selected_key = npz_file.files[0]
                else:
                    raise ValueError(
                        f"npz文件 {p} 含多个数组 {npz_file.files}，"
                        f"无法确定默认键，请使用标准键(image/arr_0)。"
                    )
            return npz_file[selected_key]
    elif suffix in ('.tif', '.tiff'):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return img
    else:
        img = cv2.imread(str(p))
        return img


class MultiModalInferenceDataset:
    """
    多模态推理专用数据集

    职责：
    - 从PairingResolver提供的MultiModalSampleSpec列表构建可迭代样本流
    - 加载RGB/X为numpy，并构造MultiModalSample
    - 严格校验：存在性、可读性、X通道数一致性
    - 统一预处理：LetterBox + 转tensor + 归一化

    设计原则：
    - 推理模式：不走label cache，不要求labels/目录存在
    - 严格输入契约：不做自动填充/降级
    - Xch任意：动态适配Router配置的通道数
    """

    def __init__(
        self,
        samples: List[Dict[str, Union[str, Path]]],
        imgsz: Union[int, tuple],
        dataset_config: Dict,
        stride: int = 32,
        pad: float = 0.5,
        verbose: bool = True
    ):
        """
        初始化推理数据集

        Args:
            samples: MultiModalSampleSpec列表（由PairingResolver提供）
            imgsz: 推理输入尺寸（int或(H,W)）
            dataset_config: 数据集配置，必须包含 'Xch' 和 'x_modality'
            stride: 模型stride（用于LetterBox对齐）
            pad: LetterBox padding比例
            verbose: 是否输出详细日志
        """
        self.samples = samples
        self.ni = len(samples)
        self.verbose = verbose

        # 推理必要配置
        # 契约驱动（Wave 4）：优先显式契约/ckpt meta，旧 Xch 按 N==2 解释
        from ultralytics.nn.mm.utils import MultiModalChannelContract
        mm_meta = dataset_config.get('mm_contract') if isinstance(dataset_config, dict) else None
        if isinstance(mm_meta, dict) and 'modality_used' in mm_meta:
            self.contract = MultiModalChannelContract.from_meta(mm_meta)
        elif isinstance(dataset_config, dict) and 'Xch' in dataset_config:
            self.contract = MultiModalChannelContract.from_meta(
                {'Xch': dataset_config['Xch'], 'Pch': dataset_config.get('Pch', 3),
                 'x_modality_name': dataset_config.get('x_modality', 'x')})
        else:
            from ultralytics.nn.mm.utils import resolve_multimodal_channel_contract
            self.contract = resolve_multimodal_channel_contract(data=dataset_config)
        self.Xch = self.contract.secondary_ch  # deprecated N==2 兼容读取
        self.x_modality = self.contract.modality_names[1] if len(self.contract.modality_names) > 1 else str(dataset_config.get('x_modality', 'unknown'))
        self.primary_modality = self.contract.modality_names[0] if self.contract.modality_names else 'rgb'
        self.primary_ch = self.contract.primary_ch

        # 图像尺寸配置
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else:
            self.imgsz = tuple(imgsz)

        self.stride = stride
        self.pad = pad

        # LetterBox配置（推理模式：不缩放放大、居中对齐）
        self.letterbox = LetterBox(
            new_shape=self.imgsz,
            auto=False,
            scale_fill=False,
            scaleup=False,  # 推理不放大
            center=True,
            stride=self.stride
        )

        # 槽位规范化（方案一）：支持槽位键 spec {'A':..,'B':..} 与旧双键 {'rgb_path','x_path'}
        self._slot_specs = []
        for s in samples:
            spec = dict(s)
            if 'A' not in spec and 'rgb_path' in spec:
                spec['A'] = spec.pop('rgb_path')
            if 'B' not in spec and 'x_path' in spec:
                spec['B'] = spec.pop('x_path')
            self._slot_specs.append(spec)

        # 兼容文件路径列表（日志/索引用）
        self.rgb_files = [str(s.get('A')) if s.get('A') is not None else '' for s in self._slot_specs]
        self.x_files = [str(s.get('B')) if s.get('B') is not None else '' for s in self._slot_specs]

        if self.verbose:
            LOGGER.info(f"MultiModalInferenceDataset 初始化完成:")
            LOGGER.info(f"  样本数: {self.ni}")
            LOGGER.info(f"  推理尺寸: {self.imgsz}")
            LOGGER.info(f"  X模态类型: {self.x_modality}")
            LOGGER.info(f"  X模态通道数: {self.Xch}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.ni

    def __getitem__(self, index: int) -> Dict:
        """获取单个多模态样本（N 槽位；精度探测前置 + fill_kind 填充）。

        Returns:
            MultiModalSample字典：
                - id / paths / orig_imgs: 槽位键 {'A': ..., 'B': ..., ...}
                - meta: 含 modality_names/channels/precision_contract 等
                - im: tensor [1, fused_ch, H, W]
        """
        spec = self._slot_specs[index]
        slot_imgs, anchor_slot = self._load_sample_images(spec)
        return self._build_sample(spec, slot_imgs, anchor_slot, index=index)

    def _load_sample_images(self, spec: Dict) -> tuple[Dict[str, np.ndarray], str | None]:
        """读取样本规格中的各槽位原图，返回 {槽位: 原图} 与锚定槽位。

        只负责读取与 numpy 维度规范化；空间对齐、精度探测、letterbox、tensorize 均交给
        ``_build_sample``，供帧目录与视频序列共同复用。
        """
        slots = list(self.contract.slots)
        slot_imgs: Dict[str, np.ndarray] = {}
        anchor_slot = next((s for s in slots if spec.get(s) is not None), None)
        for slot in slots:
            path = spec.get(slot)
            if path is None:
                continue
            loader = _load_primary_image if slot == slots[0] else self._load_x_modality
            img = loader(Path(path))
            if img is None:
                raise ValueError(f"无法读取槽位 {slot} 图像: {path}")
            if img.ndim == 2:
                img = img[..., None]
            slot_imgs[slot] = img
        return slot_imgs, anchor_slot

    def _build_sample(
        self,
        spec: Dict,
        slot_imgs: Dict[str, np.ndarray],
        anchor_slot: str | None,
        index: int = 0,
    ) -> Dict:
        """由已读取的各槽位原图构造规范化样本（对齐、精度探测、letterbox、填充、拼接）。"""
        contract = self.contract
        slots = list(contract.slots)
        slot_imgs = dict(slot_imgs)
        ratio_pad = (1.0, (0.0, 0.0))
        ori_shape = self.imgsz

        # ---- 1) 对齐：以锚定槽位原图为基准 ----
        if anchor_slot is not None:
            anchor_img = slot_imgs[anchor_slot]
            if anchor_img.ndim == 2:
                anchor_img = anchor_img[..., None]
                slot_imgs[anchor_slot] = anchor_img
            ori_shape = anchor_img.shape[:2]
            for slot in slots:
                img = slot_imgs.get(slot)
                if img is None or slot == anchor_slot:
                    continue
                if img.ndim == 2:
                    img = img[..., None]
                slot_imgs[slot] = align_and_validate_x(
                    img, anchor_img, expected_xch=contract.channels_of(slot)
                )

        # ---- 2) 精度契约前置探测（修 #4：按模态名逐槽，先于 tensorize）----
        from ultralytics.data.multimodal.precision_contract import inspect_modality_precision, LEGACY
        precision_contract: Dict[str, str] = {}
        inspections: Dict[str, dict] = {}
        for slot in slots:
            name = contract.modality_names[slots.index(slot)]
            rep = inspect_modality_precision(spec.get(slot), slot_imgs.get(slot), modality=name)
            precision_contract[name] = rep["precision"]
            inspections[name] = rep
        precision_report = {"mode": "single_point", "contracts": precision_contract,
                            "inspections": inspections}

        # ---- 3) LetterBox + 按精度 tensorize ----
        slot_tensors: Dict[str, torch.Tensor] = {}
        for slot in slots:
            ch = contract.channels_of(slot)
            name = contract.modality_names[slots.index(slot)]
            img = slot_imgs.get(slot)
            if img is not None:
                img_lb, rp = letterbox_with_ratio_pad(self.letterbox, img)
                if slot == anchor_slot:
                    ratio_pad = rp
                precision = precision_contract.get(name, LEGACY)
                if img_lb.shape[2] == 3 and contract.fill_kind(name=name) == "visible":
                    slot_tensors[slot] = to_tensor_rgb(img_lb, precision=precision)
                else:
                    slot_tensors[slot] = to_tensor_x(img_lb, precision=precision)
            else:
                # ---- 4) 缺失槽按 fill_kind 填充（修 #5：与训练语义一致）----
                H, W = self.imgsz
                slot_tensors[slot] = self._fill_missing_slot(name, ch, H, W)

        # ---- 5) 拼接 fused ----
        im = torch.cat([slot_tensors[s] for s in slots], dim=0).unsqueeze(0)
        imgsz_hw = tuple(im.shape[2:4])

        meta = {
            "modality_names": list(contract.modality_names),
            "channels": list(contract.modality_channels),
            "slots": list(slots),
            "x_modality": self.x_modality,       # deprecated
            "xch": self.Xch,                     # deprecated
            "ori_shape": ori_shape,
            "imgsz": imgsz_hw,
            "ratio_pad": ratio_pad,
            "precision_contract": precision_contract,
            "precision_report": precision_report,
        }
        for key in ("seq", "frame", "fps"):
            if key in spec:
                meta[key] = spec[key]

        return {
            "id": spec.get("id", f"sample_{index:03d}"),
            "paths": {slot: spec.get(slot) for slot in slots},
            "orig_imgs": {slot: slot_imgs.get(slot) for slot in slots},
            "meta": meta,
            "im": im,
        }

    def _fill_missing_slot(self, name: str, ch: int, H: int, W: int) -> torch.Tensor:
        """缺失槽填充：张量已归一化到 0~1，visible 槽填 114/255≈0.447（与训练 uint8 填 114 再 /255 一致），其余 0。

        边界说明：训练侧对 high_precision 整型 RGB 缺失填 0（augment._resolve_rgb_fill_value），
        推理侧缺失槽无图、探测不到精度，统一填 0.447——可接受差异。
        训练侧 fill_kind 支持 data.yaml modality_fill 覆盖；推理侧无 data_cfg，
        如需覆盖须写入 mm_contract.to_meta()（当前未支持）。
        """
        if self.contract.fill_kind(name=name) == "visible":
            return torch.full([ch, H, W], 114.0 / 255.0, dtype=torch.float32)
        return torch.zeros([ch, H, W], dtype=torch.float32)

    def _load_x_modality(self, x_path: Path) -> np.ndarray:
        """
        加载X模态图像（支持多种格式）

        Args:
            x_path: X模态图像路径

        Returns:
            X模态图像数组（H×W 或 H×W×C）

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 无法读取文件
        """
        if not x_path.exists():
            raise FileNotFoundError(f"X模态文件不存在: {x_path}")

        # 根据扩展名选择加载方式
        suffix = x_path.suffix.lower()

        try:
            if suffix == '.npy':
                # NumPy数组文件
                x_img = np.load(x_path)

            elif suffix == '.npz':
                # 压缩NumPy数组文件
                with np.load(x_path) as npz_file:
                    # 按优先级查找键
                    preferred_keys = ('image', 'arr_0', 'array', 'data')
                    selected_key = next(
                        (k for k in preferred_keys if k in npz_file.files),
                        None
                    )

                    if selected_key is None:
                        if len(npz_file.files) == 1:
                            selected_key = npz_file.files[0]
                        else:
                            raise ValueError(
                                f"npz文件含多个数组 {npz_file.files}，"
                                f"无法确定默认键，请使用标准键(image/arr_0)"
                            )

                    x_img = npz_file[selected_key]

            elif suffix in {'.tiff', '.tif'}:
                # TIFF文件（支持多通道，如16位深度图）
                x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

            else:
                # 标准图像文件（png/jpg/bmp等）
                x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

            if x_img is None:
                raise ValueError(f"无法读取X模态图像: {x_path}")

            # Legacy 分支: 返回原始数组，归一化在 tensorize 阶段按 contract 处理。
            return x_img

        except Exception as e:
            raise ValueError(f"加载X模态图像失败: {x_path}\n错误: {e}")

    def __iter__(self):
        """迭代器接口"""
        for i in range(len(self)):
            yield self[i]
