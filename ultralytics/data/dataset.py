# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from ultralytics.nn.mm.utils import get_modality_path_from_data, resolve_modality_pair, is_rgb_modality

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .multimodal.image_io import MultiModalImageIOMixin
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
    resolve_depth_gt_dir,
    img2depth_paths,
    load_depth,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return dictionary of labels for YOLO training.
        build_transforms: Build and append transforms to the list.
        close_mosaic: Set mosaic, copy_paste and mixup options to 0.0 and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data["channels"], **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        from ultralytics.utils.timedebug import add_worker_timing, merge_worker_stats

        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order

        has_timedebug = any(b.get("_timedebug_enabled", False) for b in batch)

        if has_timedebug:
            t0 = time.perf_counter()
            sample_stats_list = [b.pop("_timedebug_worker_stats", {}) for b in batch]
            for b in batch:
                b.pop("_timedebug_enabled", None)
        else:
            for b in batch:
                b.pop("_timedebug_worker_stats", None)
                b.pop("_timedebug_enabled", None)

        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

        if has_timedebug:
            merged_stats = merge_worker_stats(sample_stats_list)
            add_worker_timing(merged_stats, "batch.collate.total", time.perf_counter() - t0)
            new_batch["_timedebug_worker_stats"] = merged_stats
            new_batch["_timedebug_worker_meta"] = {"ready_abs": time.perf_counter()}
            new_batch["_timedebug_enabled"] = True

        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (Set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


def auto_detect_modality_channels(dataset_root, modality_dir, split=None, n_samples=10):
    """Sample images to detect modality channel count（fail-fast）。

    - split 依次尝试 ``train``/``val``（split=None 时）；显式 split 只查该 split
    - 目录不存在或无图：报错（不再静默返回 3）
    - 采样通道数不一致：报错（不再取 max）
    """
    dataset_root = Path(dataset_root) if not isinstance(dataset_root, Path) else dataset_root
    splits = [split] if split else ["train", "val"]
    img_dir = None
    for sp in splits:
        candidate = dataset_root / modality_dir / sp
        if candidate.exists():
            img_dir = candidate
            break
    if img_dir is None:
        raise FileNotFoundError(
            f"MultiModal: 模态目录不存在: {dataset_root / modality_dir}（尝试 split: {splits}）。"
            f"请检查 data.yaml 的 modality 映射与数据集路径。"
        )
    all_files = [f for f in img_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    if not all_files:
        raise FileNotFoundError(f"MultiModal: 模态目录无图像: {img_dir}")
    samples = random.sample(all_files, min(n_samples, len(all_files)))
    channels = set()
    for p in samples:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            ch = img.shape[2] if img.ndim == 3 else 1
            channels.add(ch)
    if len(channels) == 1:
        return channels.pop()
    raise ValueError(
        f"MultiModal: 模态目录 {modality_dir} 内通道数不一致: {sorted(channels)}。"
        f"请拆分目录或在 data.yaml 的 modality_ch 中显式声明。"
    )


class YOLOMultiModalImageDataset(YOLODataset):
    """
    多模态YOLO数据集类 - 支持任意主模态+附模态图像的目标检测数据集

    此类扩展了标准的YOLODataset，添加了对多模态图像（RGB+X）的支持。
    支持的X模态包括：深度图、热红外图、LiDAR点云图等。

    核心特性：
    - RGB+X双模态图像加载和处理
    - 6通道图像构建（RGB 3通道 + X模态 3通道）
    - 智能多模态缓存系统（内存/磁盘）
    - 自体模态生成支持（当X模态缺失时）
    - 与多模态数据增强的无缝集成
    - 高效的索引管理和验证

    数据组织结构：
    ```
    dataset/
    ├── images/
    │   ├── rgb/          # RGB图像目录
    │   └── depth/        # X模态图像目录（如深度图）
    └── labels/           # 标注文件目录
    ```

    Attributes:
        x_modality (str): X模态类型，如'depth', 'thermal', 'lidar'等
        x_modality_dir (str): X模态图像目录名
        x_modality_suffix (str): X模态图像文件后缀
        enable_self_modal_generation (bool): 是否启用自体模态生成
        x_ims (list): X模态图像缓存列表
        x_im_hw0 (list): X模态图像原始尺寸缓存
        x_im_hw (list): X模态图像调整后尺寸缓存

    Methods:
        __getitem__: 获取多模态数据样本（6通道图像+标签）
        get_valid_indices: 获取有完整多模态数据的有效索引
        load_multimodal_image: 加载RGB+X模态图像并构建6通道图像
        cache_images: 缓存RGB和X模态图像到内存或磁盘

    Examples:
        >>> # 基本使用
        >>> dataset = YOLOMultiModalImageDataset(
        ...     img_path="dataset/images/rgb",
        ...     x_modality="depth",
        ...     x_modality_dir="depth"
        ... )
        >>> sample = dataset[0]  # 返回6通道图像和标签
        >>> print(sample["img"].shape)  # (6, H, W) - RGB+Depth

        >>> # 启用自体模态生成
        >>> dataset = YOLOMultiModalImageDataset(
        ...     img_path="dataset/images/rgb",
        ...     x_modality="thermal",
        ...     enable_self_modal_generation=True
        ... )
    """

    def __init__(
        self,
        *args,
        x_modality="depth",
        x_modality_dir=None,
        x_modality_suffix=None,
        enable_self_modal_generation=False,
        precision_report_dir=None,
        mm_contract=None,
        **kwargs
    ):
        """
        初始化多模态YOLO数据集（N 模态，契约驱动，Wave 3）。

        Args:
            mm_contract (MultiModalChannelContract, optional): trainer 传入的同一契约对象；
                缺省时 super().__init__ 后从 self.data 解析一次。
            x_modality / x_modality_dir / x_modality_suffix: deprecated（Wave 4 删除），
                仅保留签名兼容，不再用于通道推导。
            enable_self_modal_generation (bool): 是否启用自体模态生成
            precision_report_dir: 精度契约报告目录
            *args, **kwargs: 传递给父类
        """
        self._pending_contract = mm_contract
        self._mm_paths_ready = False
        self._cache_pending = False
        self._ram_budget_checked = False

        # deprecated 属性（Wave 4 删除）
        self.primary_modality = None
        self.secondary_modality = x_modality
        self.x_modality = x_modality
        self.x_modality_dir = x_modality_dir
        self.x_modality_suffix = x_modality_suffix
        self.enable_self_modal_generation = enable_self_modal_generation
        self.precision_report_dir = precision_report_dir
        self.primary_ch = None
        self.secondary_ch = None
        self.precision_contract = None  # _init_mm_contract 后按模态名填充
        self.precision_report = None
        self.precision_report_path = None
        # 缺失容忍（CVPR 构建方案 §3.3）：true 时缺模态样本不过滤，零张量 + mm_avail=0
        self.mm_allow_missing = False
        # 按槽位缓存（Wave 3.2）
        self.contract = None
        self._mod_ims = {}
        self._mod_im_hw0 = {}
        self._mod_im_hw = {}
        self._mod_npy_files = {}
        self._mod_paths = {}
        self._mod_buffers = {}
        self.mm_cache_root_name = ".ultralytics_mm_cache"

        # 兼容旧字段（val/plotter 仍读；Wave 4 迁移）
        self.x_ims = None
        self.x_im_hw0 = None
        self.x_im_hw = None
        self.x_npy_files = None
        self._x_buffer = []
        self._x_paths: List[Optional[str]] = []
        self._modality_paths: List[List[Optional[str]]] = []

        # 提前解析 primary_modality_dir（get_labels 在 super().__init__ 中需要）
        data_dict = kwargs.get('data', None)
        if isinstance(data_dict, dict):
            from ultralytics.nn.mm.utils import (
                get_configured_modality_tokens,
                get_modality_path_from_data,
                resolve_multimodal_channel_contract,
            )

            tokens = get_configured_modality_tokens(data_dict)
            primary = tokens[0] if tokens else 'rgb'
            self.primary_modality_dir = get_modality_path_from_data(data_dict, primary)
            # 提前解析契约：BaseDataset.__init__ 内的 RAM 预算估算需要全部槽位
            self.contract = mm_contract or resolve_multimodal_channel_contract(data=data_dict)
        else:
            self.primary_modality_dir = 'images'
            self.contract = mm_contract

        # 调用父类初始化
        super().__init__(*args, **kwargs)

        # 缺失容忍开关从 data.yaml 读取（构建方案 §3.3）
        _dd = getattr(self, "data", None)
        if isinstance(_dd, dict):
            self.mm_allow_missing = bool(_dd.get("mm_allow_missing", False))

        # 契约解析 + 按槽位初始化（Wave 3.1）
        self._init_mm_contract()

        # 启动阶段基于 split 解析精度契约并过滤不匹配样本（仅一次）
        self._initialize_precision_contract()

        # 验证多模态数据完整性（按槽位）
        self._validate_multimodal_data()

        # RAM/disk 缓存必须在精度过滤之后建立：im_files/labels/ni 与 _mod_paths 已对齐
        if self._cache_pending:
            if self.cache == "ram" and not self._ram_budget_checked:
                self.check_cache_ram()
            self.cache_images()

        # 契约化初始化日志（Wave 3.5）
        modal_info = "+".join(
            f"{name}({slot},{ch}ch)" for slot, name, ch in zip(self.contract.slots, self.contract.modality_names, self.contract.modality_channels)
        )
        LOGGER.info(f"MultiModal: {modal_info} fused={self.contract.fused_ch}ch 数据集已初始化")
        LOGGER.info(f"MultiModal: 总样本数={self.ni}")

    def _init_mm_contract(self):
        """解析/接收通道契约并初始化按槽位的路径与缓存结构（Wave 3.1/3.2）。"""
        from ultralytics.nn.mm.utils import resolve_multimodal_channel_contract

        contract = self._pending_contract or self.contract or resolve_multimodal_channel_contract(
            data=getattr(self, 'data', None)
        )
        self.contract = contract
        self.primary_ch = contract.primary_ch
        self.secondary_ch = contract.secondary_ch
        # 显示名对齐契约
        if contract.modality_names:
            self.primary_modality = contract.modality_names[0]
        if len(contract.modality_names) > 1:
            self.secondary_modality = contract.modality_names[1]

        # x_modality 兼容属性指向槽位 B
        self.x_modality = contract.modality_names[1] if len(contract.modality_names) > 1 else self.x_modality

        # precision contract 按模态名（Wave 3.6）；旧 rgb/x 两键语义映射到槽位 A/B
        if self.precision_contract is None or set(self.precision_contract) == {"rgb", "x"}:
            legacy = {"rgb": "legacy", "x": "legacy"}
            if isinstance(self.precision_contract, dict):
                for k in ("rgb", "x"):
                    if k in self.precision_contract:
                        legacy[k] = self.precision_contract[k]
            self.precision_contract = {}
            for slot, name in zip(contract.slots, contract.modality_names):
                old_key = "rgb" if slot == contract.slots[0] else ("x" if slot == contract.slots[1] else None)
                self.precision_contract[name] = legacy.get(old_key, "legacy") if old_key else "legacy"

        # 按槽位预计算路径与缓存骨架
        for slot, name in zip(contract.slots, contract.modality_names):
            self._mod_paths[slot] = [None] * self.ni
            self._mod_buffers[slot] = []
        self._precompute_mod_paths()
        self._mm_paths_ready = True
        # 兼容：槽位 B 路径镜像到 _x_paths（precision contract / 校验读）
        if len(contract.slots) > 1:
            self._x_paths = list(self._mod_paths.get('B', []))
            slot_b_dir = self._mod_dir_of(contract.modality_names[1])
            self._secondary_modality_dir = slot_b_dir if hasattr(self, '_secondary_modality_dir') else slot_b_dir
            if getattr(self, 'x_modality_dir', None) is None:
                self.x_modality_dir = slot_b_dir

    def _mod_dir_of(self, modality_name: str) -> str:
        """模态名 → 目录名（data.yaml modality 映射，缺省 images_<name>）。"""
        from ultralytics.nn.mm.utils import get_modality_path_from_data
        return get_modality_path_from_data(getattr(self, 'data', None), modality_name)

    def _precompute_mod_paths(self) -> None:
        """按槽位预计算模态路径，与 im_files 1:1 对齐（Wave 3.2）。"""
        if not getattr(self, 'im_files', None):
            return
        for slot, name in zip(self.contract.slots[1:], self.contract.modality_names[1:]):
            paths = []
            for i in range(self.ni):
                try:
                    paths.append(self._find_corresponding_x_image(self.im_files[i], modality_name=name))
                except Exception:
                    paths.append(None)
            self._mod_paths[slot] = paths

    def _initialize_precision_contract(self) -> None:
        """Resolve dataset-run precision contract and filter outlier samples before dataloader usage."""
        from ultralytics.data.multimodal.precision_contract import (
            filter_dataset_by_contract,
            log_dataset_precision_summary,
            resolve_precision_contract,
        )

        if not getattr(self, "im_files", None):
            return

        report = resolve_precision_contract(self, mode="dataset_run")
        keep_indices, filtered = filter_dataset_by_contract(self, report["contracts"])
        split = Path(self.im_files[0]).parent.name if self.im_files else "unknown"
        report["split"] = split
        report["filtered_count"] = len(filtered)
        report["filtered_examples"] = filtered

        if keep_indices and len(keep_indices) < len(self.im_files):
            self._apply_filtered_indices(keep_indices)
            report["filter_applied"] = True
        elif not keep_indices and filtered:
            LOGGER.warning(
                "MultiModal precision contract would filter all %d samples for split=%s, keeping original dataset",
                len(self.im_files),
                split,
            )
            report["filter_applied"] = False
            report["filtered_count"] = 0
            report["filtered_examples"] = []
        else:
            report["filter_applied"] = False

        self.precision_contract = dict(report["contracts"])
        self.precision_report = report
        self.precision_report_path = None
        log_dataset_precision_summary(report, None)

    def _apply_filtered_indices(self, keep_indices: List[int]) -> None:
        """Reindex dataset state after contract-based outlier filtering."""
        old_ni = len(self.im_files)
        self.im_files = [self.im_files[i] for i in keep_indices]
        self.labels = [self.labels[i] for i in keep_indices]
        self.ni = len(self.labels)
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.x_ims, self.x_im_hw0, self.x_im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self._x_buffer = []
        self._x_paths = [self._x_paths[i] for i in keep_indices] if self._x_paths else []
        # 槽位缓存与 im_files 必须保持 1:1；否则过滤后 RGB/IR 会错配。
        for slot, paths in list(self._mod_paths.items()):
            if isinstance(paths, list) and len(paths) == old_ni:
                self._mod_paths[slot] = [paths[i] for i in keep_indices]
        for cache in (self._mod_ims, self._mod_im_hw0, self._mod_im_hw, self._mod_npy_files):
            for slot, values in list(cache.items()):
                if isinstance(values, list) and len(values) == old_ni:
                    cache[slot] = [values[i] for i in keep_indices]
        for slot in list(self._mod_buffers):
            self._mod_buffers[slot] = []
        if hasattr(self, "_valid_indices_cache"):
            del self._valid_indices_cache
        self._refresh_x_npy_files()
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        if self.rect and self.ni:
            for i, label in enumerate(self.labels):
                if "shape" not in label:
                    try:
                        _, shape0, _ = self.load_image(i, rect_mode=False)
                    except Exception:
                        shape0 = (self.imgsz, self.imgsz)
                    label["shape"] = shape0
            self.set_rectangle()

    # -- backward compat properties for x_modality / x_modality_dir --

    @property
    def x_modality(self):
        """Backward compat: returns secondary_modality."""
        return getattr(self, '_secondary_modality', getattr(self, '_x_modality', 'unknown'))

    @x_modality.setter
    def x_modality(self, value):
        self._secondary_modality = value

    @property
    def x_modality_dir(self):
        """Backward compat: returns secondary_modality_dir."""
        return getattr(self, '_secondary_modality_dir', getattr(self, '_x_modality_dir', 'images'))

    @x_modality_dir.setter
    def x_modality_dir(self, value):
        self._secondary_modality_dir = value

    def _filter_legacy_x_cache_candidates(self, im_files: List[str]) -> List[str]:
        """Remove legacy X cache files accidentally written into primary image directories."""
        from ultralytics.data.utils import ARRAY_FORMATS, IMG_FORMATS

        x_modality = str(getattr(self, "x_modality", "") or "")
        if not x_modality:
            return im_files

        legacy_suffix = f"_{x_modality}"
        file_index = {
            (str(Path(f).parent), Path(f).stem, Path(f).suffix[1:].lower())
            for f in im_files
        }
        filtered = []
        removed = 0

        for f in im_files:
            p = Path(f)
            suffix = p.suffix[1:].lower()
            if suffix in ARRAY_FORMATS and p.stem.endswith(legacy_suffix):
                base_stem = p.stem[: -len(legacy_suffix)]
                has_primary_sibling = any(
                    (str(p.parent), base_stem, ext) in file_index for ext in IMG_FORMATS
                )
                if has_primary_sibling:
                    removed += 1
                    continue
            filtered.append(f)

        if removed:
            LOGGER.info(
                f"MultiModal: ignored {removed} legacy X cache files in primary image directories "
                f"(*_{x_modality}.npy/.npz). New X cache is stored under {self.mm_cache_root_name}/."
            )
        return filtered

    def get_img_files(self, img_path):
        """Override to support primary NPY/NPZ while ignoring legacy X cache pollution."""
        from ultralytics.data.utils import MM_PRIMARY_FORMATS

        im_files = super().get_img_files(img_path, allowed_formats=MM_PRIMARY_FORMATS)
        return self._filter_legacy_x_cache_candidates(im_files)

    def _img2label_path(self, img_path: Union[str, Path]) -> Path:
        """Convert image path to label path, supporting non-standard primary dirs (e.g. images_ir)."""
        img_path = Path(img_path)
        parts = list(img_path.parts)
        primary_dir = getattr(self, 'primary_modality_dir', None) or 'images'
        for i, part in enumerate(parts):
            if part == primary_dir:
                parts[i] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def get_labels(self) -> List[Dict]:
        """Override: use multimodal-aware label path resolution.

        Replaces the hardcoded ``img2label_paths`` (which assumes ``/images/``
        directory) with ``_img2label_path`` that respects ``primary_modality_dir``
        (e.g. ``images_ir`` for IR+Depth datasets).
        """
        self.label_files = [str(self._img2label_path(f)) for f in self.im_files]
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def _validate_multimodal_data(self):
        """验证多模态数据的完整性和一致性"""
        if not hasattr(self, 'im_files') or not self.im_files:
            LOGGER.warning("主模态图像文件列表为空，无法验证多模态数据")
            return

        # 检查非 A 槽位目录是否存在（契约驱动；不再读取 deprecated x_modality_dir）
        x_dir = None
        for slot in self.contract.slots[1:]:
            paths = self._mod_paths.get(slot, [])
            first_path = next((p for p in paths if p is not None), None)
            if first_path is not None:
                x_dir = Path(first_path).parent.parent  # .../images_<modality>
                break

        if x_dir is not None and not x_dir.exists():
            if self.enable_self_modal_generation:
                LOGGER.info(f"X模态目录 {x_dir} 不存在，将使用自体模态生成")
            else:
                LOGGER.debug(f"X模态目录 {x_dir} 不存在，将尝试从其他位置加载或使用自体生成")

        # 按（非 A）槽位统计完整性与容量（Wave 3.2）
        sample_size = min(len(self.im_files), 100)  # 采样检查前100个
        for idx, (slot, name) in enumerate(zip(self.contract.slots, self.contract.modality_names)):
            valid_count = 0
            if idx == 0:
                # 槽位 A：主模态即 im_files 本身
                valid_count = sum(1 for i in range(sample_size) if Path(self.im_files[i]).exists())
            else:
                paths = self._mod_paths.get(slot, [])
                for i in range(sample_size):
                    x_path = paths[i] if i < len(paths) else None
                    if x_path is not None and (Path(x_path).exists() or self.enable_self_modal_generation):
                        valid_count += 1
            valid_ratio = valid_count / sample_size if sample_size > 0 else 0
            LOGGER.info(f"MultiModal: 模态 {name}[{slot}] 完整性检查 - {valid_ratio:.1%} ({valid_count}/{sample_size})")
            if valid_ratio < 0.5 and not self.enable_self_modal_generation:
                LOGGER.warning(f"MultiModal: 模态 {name}[{slot}] 完整性较低，建议启用自体模态生成或检查数据路径")

    def _precompute_x_paths(self) -> None:
        """预计算X模态路径列表，与im_files 1:1对齐。可在_apply_filtered_indices后重建。"""
        self._x_paths = []
        for i in range(self.ni):
            try:
                rgb_path = self.im_files[i]
                self._x_paths.append(self._find_corresponding_x_image(rgb_path))
            except Exception:
                self._x_paths.append(None)

    def __getitem__(self, index):
        """
        获取多模态数据样本

        Args:
            index (int): 样本索引

        Returns:
            dict: 包含6通道多模态图像和标签的字典
                - 'img': 6通道图像张量 [6, H, W] (RGB+X)
                - 其他标签信息（bbox, cls等）

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> sample = dataset[0]
            >>> img = sample['img']  # 6通道图像
            >>> print(img.shape)  # torch.Size([6, H, W])
        """
        if not getattr(self, "timedebug_enabled", False):
            # 获取基础标签信息（使用父类方法）
            label = self.get_image_and_label(index)

            # 加载多模态图像（替换单模态RGB图像）
            multimodal_img = self.load_multimodal_image(index)
            label["img"] = multimodal_img
            label["precision_contract"] = dict(self.precision_contract)
            label["precision_report_path"] = self.precision_report_path
            label["precision_mode"] = "dataset_run"
            label["mm_contract"] = self.contract
            label["data_cfg"] = getattr(self, "data", None)
            # 槽位可用性（构建方案 §3.3）：供 trainer 拼接 availability mask
            avail = self.mm_slot_avail(index)
            label["mm_avail"] = avail
            # 应用数据变换；增广链（Mosaic/MixUp）会重建 dict 丢掉未知键，故变换后重新注入
            out = self.transforms(label)
            out["mm_avail"] = avail
            return out

        from ultralytics.utils.timedebug import add_sample_timing

        t0 = time.perf_counter()
        # 获取基础标签信息（使用父类方法）
        t_label = time.perf_counter()
        label = self.get_image_and_label(index)
        label["_timedebug_enabled"] = True
        add_sample_timing(label, "sample.get_image_and_label", time.perf_counter() - t_label)

        # 加载多模态图像（替换单模态RGB图像）
        t_load_mm = time.perf_counter()
        timing_stats = label.setdefault("_timedebug_worker_stats", {})
        try:
            multimodal_img = self.load_multimodal_image(index, timing_stats=timing_stats)
        except TypeError as e:
            if "timing_stats" not in str(e):
                raise
            multimodal_img = self.load_multimodal_image(index)
        add_sample_timing(label, "sample.load_multimodal_image", time.perf_counter() - t_load_mm)
        label["img"] = multimodal_img
        label["precision_contract"] = dict(self.precision_contract)
        label["precision_report_path"] = self.precision_report_path
        label["precision_mode"] = "dataset_run"
        label["mm_contract"] = self.contract
        label["data_cfg"] = getattr(self, "data", None)
        avail = self.mm_slot_avail(index)
        label["mm_avail"] = avail
        # 应用数据变换；增广链会重建 dict 丢掉未知键，变换后重新注入
        t_transforms = time.perf_counter()
        label = self.transforms(label)
        label["mm_avail"] = avail
        add_sample_timing(label, "sample.transforms_total", time.perf_counter() - t_transforms)
        add_sample_timing(label, "sample.total", time.perf_counter() - t0)
        return label

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        构建并返回多模态专用增强流水线（独立于通用 v8_transforms）。

        说明：
        - 多模态数据集一律走独立 mm_transforms() 链路，可复用 v8 组件；
        - 单/双模态的路由与消融由路由层负责，数据层不做区分。
        """
        from ultralytics.data.multimodal_augment import mm_transforms, mm_seg_transforms
        from ultralytics.data.augment import Compose, LetterBox, Format

        if self.augment:
            # 强制走多模态增强链；分割任务走 mm_seg_transforms
            if getattr(self, "use_segments", False):
                transforms = mm_seg_transforms(self, self.imgsz, hyp)
            else:
                transforms = mm_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        # 统一追加 Format；多模态训练不使用 bgr 概率翻转
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,  # 禁用 BGR 随机翻转，避免触动 X 通道
            )
        )
        return transforms

    def get_valid_indices(self):
        """
        获取有完整多模态数据的有效索引列表

        此方法用于与MultiModalMosaic和MultiModalMixUp数据增强兼容，
        确保增强过程中选择的图像都有完整的RGB+X模态数据。

        Returns:
            list: 有效索引列表

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> valid_indices = dataset.get_valid_indices()
            >>> print(f"有效样本数: {len(valid_indices)}")
        """
        if not hasattr(self, '_valid_indices_cache'):
            self._valid_indices_cache = []

            # 仅在主进程/主 worker 打印日志，避免多 worker 噪声
            log_enabled = True
            try:
                from torch.utils.data import get_worker_info
                wi = get_worker_info()
                log_enabled = (wi is None) or (wi.id == 0)
            except Exception:
                log_enabled = True

            if log_enabled:
                LOGGER.info("MultiModal: 正在扫描有效的多模态索引...")

            non_a_slots = self.contract.slots[1:] if getattr(self, "contract", None) is not None else ()
            for i in range(self.ni):
                # 检查槽位 A（主模态）图像是否存在
                if not Path(self.im_files[i]).exists():
                    continue

                # 检查所有非 A 槽位；mm_allow_missing 时缺模态不跳过（零张量 + avail=0）
                valid = True
                if not self.mm_allow_missing:
                    for slot in non_a_slots:
                        paths = self._mod_paths.get(slot, [])
                        x_path = paths[i] if i < len(paths) else None
                        if x_path is None or not (Path(x_path).exists() or self.enable_self_modal_generation):
                            valid = False
                            break
                if valid:
                    self._valid_indices_cache.append(i)

            valid_count = len(self._valid_indices_cache)
            valid_ratio = valid_count / self.ni if self.ni > 0 else 0

            if log_enabled:
                LOGGER.info(f"✅ MultiModal: 发现 {valid_count}/{self.ni} ({valid_ratio:.1%}) 个有效多模态样本")

                if valid_count == 0:
                    LOGGER.error("未发现任何有效的多模态样本，请检查数据路径和配置")
                elif valid_ratio < 0.8:
                    LOGGER.warning(f"有效多模态样本比例较低 ({valid_ratio:.1%})，建议检查数据完整性")

        return self._valid_indices_cache

    def mm_slot_avail(self, i):
        """样本 i 的各槽位可用性向量 [M]（float32）：文件存在=1，缺失=0（构建方案 §3.3）。

        A 槽以 im_files 为准；其余槽以 _mod_paths 预计算路径为准。
        """
        avail = []
        for slot_idx, slot in enumerate(self.contract.slots):
            if slot_idx == 0:
                avail.append(1.0 if Path(self.im_files[i]).exists() else 0.0)
            else:
                paths = self._mod_paths.get(slot, [])
                p = paths[i] if i < len(paths) else None
                avail.append(1.0 if (p is not None and Path(p).exists()) else 0.0)
        return np.asarray(avail, dtype=np.float32)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate 多模态样本：额外 stack mm_avail 成 [B,M]（构建方案 §3.3）。

        缺此覆盖时 batch 里没有 mm_avail，模型的 availability mask 恒为默认全 1，
        导致子集掉线训练/评测完全失效。
        """
        new_batch = YOLODataset.collate_fn(batch)
        avail_list = [b.get("mm_avail") for b in batch]
        if avail_list and all(a is not None for a in avail_list):
            new_batch["mm_avail"] = torch.stack(
                [torch.as_tensor(np.asarray(a), dtype=torch.float32) for a in avail_list], 0
            )
        # 合成退化标签（健康训练）：[M,2] → [B,M,2]；缺键时不写入，保持旧行为
        corr_list = [b.get("mm_corruption") for b in batch]
        if corr_list and all(c is not None for c in corr_list):
            new_batch["mm_corruption"] = torch.stack(
                [torch.as_tensor(np.asarray(c), dtype=torch.float32) for c in corr_list], 0
            )
        return new_batch

    def load_multimodal_image(self, i, timing_stats=None):
        """循环加载 N 个模态的图像并拼接（三模态及以上版本）。
        
        支持 RGB + Event + Thermal 等 N 个模态的循环加载。
        按 modality_used 顺序加载各模态并在通道维度拼接。
        
        Args:
            i (int): 图像索引
            timing_stats (dict | None): worker timing stats dict，用于子步骤计时。
        
        Returns:
            np.ndarray: N 通道拼接图像 [H, W, sum(channels)]
        
        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)  # 三模态数据集
            >>> img_9ch = dataset.load_multimodal_image(0)  # [H, W, 9] for RGB+Event+Thermal
        """
        from ultralytics.utils.timedebug import add_worker_timing
        import time
        
        # 模态列表只来自数据集持有的契约（Wave 5 清理；裸读旧字段会拿到 'unknown' 默认值）
        modality_used = list(self.contract.modality_names)
        
        # 循环加载各模态
        t_start = time.perf_counter()
        images = []
        h, w = None, None
        
        for slot_idx, mod_name in enumerate(modality_used):
            t_mod = time.perf_counter()
        
            if slot_idx == 0:
                # 槽位 A（主模态 RGB）用父类 load_image 加载
                img, (h0, w0), (h, w) = self.load_image(i)
                # 槽位 A 按契约校正通道（如 depth 单模态 1ch：load_image 固定读成 3ch BGR）
                ch_a = self.contract.channels_of(self.contract.slots[0])
                if img.ndim == 2:
                    img = img[..., None]
                if img.shape[2] != ch_a:
                    if ch_a == 1:
                        img = img[..., :1]
                    else:
                        raise ValueError(
                            f"槽位 A({mod_name}) 通道 {img.shape[2]} != 契约 {ch_a}；"
                            f"契约 {self.contract}"
                        )
                LOGGER.debug(f"[MM] 槽位 {slot_idx} ({mod_name}): 加载成功, shape={img.shape}, 来源=load_image()")
            else:
                # 其他槽位用 _load_modality_cached
                # 目标尺寸为 RGB 的尺寸
                try:
                    img, _, _ = self._load_modality_cached(i, mod_name, target_shape=(h, w))
                    LOGGER.debug(f"[MM] 槽位 {slot_idx} ({mod_name}): 加载成功, shape={img.shape}")
                except Exception as e:
                    LOGGER.error(f"[MM] 槽位 {slot_idx} ({mod_name}): 加载失败: {e}")
                    if self.enable_self_modal_generation:
                        LOGGER.debug(f"[MM] 槽位 {slot_idx} ({mod_name}): 使用自体生成")
                        img = self._generate_self_modal(images[0])  # 基于 RGB 生成
                    elif self.mm_allow_missing and h is not None and w is not None:
                        # 缺失容忍：正确通道数的零张量（构建方案 §3.3）
                        ch = self.contract.channels_of(self.contract.slots[slot_idx])
                        img = np.zeros((h, w, ch), dtype=np.uint8)
                    else:
                        raise
        
            # 确保尺寸一致
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
        
            images.append(img)
            add_worker_timing(timing_stats, f"event.{mod_name}.load", time.perf_counter() - t_mod)
        
        # 拼接所有模态
        t_concat = time.perf_counter()
        multimodal_img = np.concatenate(images, axis=2)
        add_worker_timing(timing_stats, "event.mm.concat", time.perf_counter() - t_concat)
        add_worker_timing(timing_stats, "event.mm.total", time.perf_counter() - t_start)
        
        return multimodal_img

    def _load_multimodal_components(self, i, timing_stats=None):
        """返回按槽位的图像列表 list[np.ndarray]（Wave 3.4，Mosaic donor 用 `_mm_parts`）。

        - 槽位 A 用父类 load_image
        - 其余槽位用 _load_modality_cached（真 cache + 通道校验）
        - 不再补 3ch 零图占位；自体生成按槽位各自应用并校验通道
        """
        from ultralytics.utils.timedebug import add_worker_timing
        import time

        t_start = time.perf_counter()

        parts = []
        a_img, (h0, w0), (h, w) = self.load_image(i)
        # 槽位 A 按契约校正通道（depth 单模态 1ch 等；load_image 固定 3ch BGR）
        ch_a = self.contract.channels_of(self.contract.slots[0])
        if a_img.ndim == 2:
            a_img = a_img[..., None]
        if a_img.shape[2] != ch_a:
            if ch_a == 1:
                a_img = a_img[..., :1]
            else:
                raise ValueError(f"槽位 A({self.contract.modality_names[0]}) 通道 {a_img.shape[2]} != 契约 {ch_a}")
        parts.append(a_img)

        for slot, name in zip(self.contract.slots[1:], self.contract.modality_names[1:]):
            t_mod = time.perf_counter()
            try:
                img, _, _ = self._load_modality_cached(i, name, target_shape=(h, w))
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                if img.ndim == 2:
                    img = img[..., None]
            except Exception as e:
                if self.enable_self_modal_generation:
                    img = self._generate_self_modal(a_img)
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))
                    expected = self.contract.channels_of(slot)
                    if img.shape[2] != expected:
                        raise ValueError(
                            f"自体生成模态 {name} 通道 {img.shape[2]} != 契约 {expected}（槽位 {slot}）"
                        ) from e
                elif self.mm_allow_missing:
                    # 缺失容忍：正确通道数零张量（构建方案 §3.3）
                    expected = self.contract.channels_of(slot)
                    img = np.zeros((h, w, expected), dtype=np.uint8)
                else:
                    raise
            parts.append(img)
            add_worker_timing(timing_stats, f"event.{name}.load", time.perf_counter() - t_mod)

        add_worker_timing(timing_stats, "event.mm_components.total", time.perf_counter() - t_start)
        return parts

    def _find_corresponding_x_image(self, rgb_path, modality_name=None):
        """
        根据主模态图像路径找到对应模态的图像路径（N 模态，Wave 3）。

        Args:
            rgb_path (str): 主模态（槽位 A）图像路径
            modality_name (str, optional): 目标模态名；缺省用 self.x_modality（槽位 B）

        Returns:
            str: 目标模态图像路径

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(x_modality_dir="images_ir")
            >>> rgb_path = "/dataset/images/train/img001.jpg"
            >>> x_path = dataset._find_corresponding_x_image(rgb_path, modality_name="ir")
            >>> print(x_path)  # "/dataset/images_ir/train/img001.jpg"
        """
        rgb_path = Path(rgb_path)
        if modality_name is not None:
            mod_dir_name = self._mod_dir_of(modality_name)
        else:
            mod_dir_name = self.x_modality_dir or self._mod_dir_of(self.x_modality)

        # 构建目标模态图像路径
        # 保持相同的子目录结构 (train/val/test)，只替换基础模态目录
        dataset_root = rgb_path.parent.parent.parent  # /dataset (跳过images和train)
        split_dir = rgb_path.parent.name              # train/val/test
        x_dir = dataset_root / mod_dir_name / split_dir

        # 处理文件后缀
        if self.x_modality_suffix:
            x_filename = rgb_path.stem + self.x_modality_suffix + rgb_path.suffix
        else:
            x_filename = rgb_path.name

        x_path = x_dir / x_filename

        # 如果指定路径不存在，尝试常见的文件扩展名（包含npy/npz特征文件）
        if not x_path.exists():
            common_extensions = ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            for ext in common_extensions:
                test_path = x_dir / (rgb_path.stem + ext)
                if test_path.exists():
                    return str(test_path)

        return str(x_path)

    def _load_x_modality(self, x_path):
        """
        加载X模态图像

        Args:
            x_path (str): X模态图像路径

        Returns:
            np.ndarray: X模态图像数组

        Raises:
            FileNotFoundError: 当X模态图像文件不存在时
        """
        x_path = Path(x_path)

        if not x_path.exists():
            raise FileNotFoundError(f"X模态图像不存在: {x_path}")

        # 根据文件扩展名选择加载方式
        if x_path.suffix.lower() in ['.npy']:
            # NumPy数组文件
            x_img = np.load(x_path)
        elif x_path.suffix.lower() in ['.npz']:
            # 多数组压缩文件，按约定顺序择优取出
            with np.load(x_path) as npz_file:
                preferred_keys = ('image', 'arr_0', 'array', 'data')
                selected_key = next((k for k in preferred_keys if k in npz_file.files), None)

                if selected_key is None:
                    if len(npz_file.files) == 1:
                        selected_key = npz_file.files[0]
                    else:
                        raise ValueError(
                            f"npz文件 {x_path} 含多个数组 {npz_file.files}，无法确定默认键，请显式使用标准键(image/arr_0)。"
                        )

                x_img = npz_file[selected_key]
        elif x_path.suffix.lower() in ['.tiff', '.tif']:
            # TIFF文件（常用于深度图）
            x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)
        else:
            # 标准图像文件
            x_img = cv2.imread(str(x_path))

        if x_img is None:
            raise ValueError(f"无法读取X模态图像: {x_path}")

        # Legacy 分支: 返回原始数组，归一化在 preprocess 阶段按 contract 处理。
        return x_img


    def _load_modality_cached(self, i, modality_name, target_shape=None):
        """循环加载指定模态的图像（N 模态扩展版本）。
        
        Args:
            i (int): 图像索引
            modality_name (str): 模态名称（如 'event', 'thermal'）
            target_shape (tuple, optional): 目标尺寸 (H, W)
        
        Returns:
            tuple: (img, hw0, hw)
        """
        import cv2
        from ultralytics.utils import LOGGER

        # 契约通道校验目标
        expected_ch = None
        if self.contract is not None:
            try:
                expected_ch = self.contract.channels_of(self.contract.slot_of(modality_name))
            except ValueError:
                expected_ch = None

        # 先查按槽位内存缓存（Wave 3.3：真正读写 cache）
        slot = self.contract.slot_of(modality_name) if self.contract else None
        if slot is not None:
            ims = self._mod_ims.get(slot)
            if ims is not None and i < len(ims) and ims[i] is not None:
                img = ims[i]
                hw0 = self._mod_im_hw0[slot][i]
                hw = self._mod_im_hw[slot][i]
                if target_shape and img.shape[:2] != target_shape:
                    img = cv2.resize(img, (target_shape[1], target_shape[0]))
                    self._cache_mod_im(slot, i, img, *hw0)
                    hw = img.shape[:2]
                return img, hw0, hw
            # 再查磁盘缓存
            npy_file = self._get_mod_npy_file(slot, i)
            if npy_file is not None and npy_file.exists():
                try:
                    img = np.load(npy_file, allow_pickle=False)
                    if img.ndim == 2:
                        img = img[..., None]
                    h0, w0 = img.shape[:2]
                    if target_shape and img.shape[:2] != target_shape:
                        img = cv2.resize(img, (target_shape[1], target_shape[0]))
                        if img.ndim == 2:
                            img = img[..., None]
                    self._cache_mod_im(slot, i, img, h0, w0)
                    return img, (h0, w0), img.shape[:2]
                except Exception as e:
                    LOGGER.warning(f"加载模态 {modality_name} 磁盘缓存失败 {npy_file}: {e}")

        # 获取模态配置
        modality_config = self.data.get('modality', {})
        modality_dir = modality_config.get(modality_name, f'images_{modality_name}')

        # 构建模态路径
        rgb_path = Path(self.im_files[i])
        dataset_root = Path(self.data.get('path', rgb_path.parent.parent.parent))
        split_dir = rgb_path.parent.name
        mod_dir = dataset_root / modality_dir / split_dir
        mod_path = mod_dir / rgb_path.name
        
        # [DEBUG] 首次加载或路径解析失败时输出日志
        _loaded_modality_cache = getattr(self, '_loaded_modality_cache', set())
        cache_key = (i, modality_name)
        if cache_key not in _loaded_modality_cache:
            LOGGER.debug(f"[MM] 加载模态: {modality_name}, i={i}, path={mod_path}")
            _loaded_modality_cache.add(cache_key)
            self._loaded_modality_cache = _loaded_modality_cache
        # 尝试常见扩展名
        if not mod_path.exists():
            common_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy', '.npz']
            for ext in common_extensions:
                test_path = mod_dir / (rgb_path.stem + ext)
                if test_path.exists():
                    mod_path = test_path
                    break
        
        # 加载图像
        if mod_path.suffix.lower() in ['.npy']:
            img = np.load(mod_path)
        elif mod_path.suffix.lower() in ['.npz']:
            with np.load(mod_path) as npz_file:
                key = 'image' if 'image' in npz_file.files else npz_file.files[0]
                img = npz_file[key]
        elif mod_path.suffix.lower() in ['.tiff', '.tif']:
            img = cv2.imread(str(mod_path), cv2.IMREAD_UNCHANGED)
        else:
            # 非 A 模态统一原样读取（Wave 3.3 / 决策 §2.2）
            img = cv2.imread(str(mod_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(f"模态 {modality_name} 图像不存在: {mod_path}")

        # 2D 补 H×W×1
        if img.ndim == 2:
            img = img[..., None]

        # 通道校验：读出通道数必须等于契约声明（决策 §2.2，fail-fast）
        got_ch = img.shape[2] if img.ndim == 3 else 1
        if expected_ch is not None and got_ch != expected_ch:
            raise ValueError(
                f"模态 {modality_name} 读出 {got_ch}ch 与契约 {expected_ch}ch 不符: {mod_path}"
            )

        h0, w0 = img.shape[:2]

        # 调整尺寸（resize 后再补 ×1）
        if target_shape and img.shape[:2] != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]))
            if img.ndim == 2:
                img = img[..., None]
            h, w = target_shape
        else:
            h, w = h0, w0

        # 写回按槽位缓存
        if slot is not None:
            self._cache_mod_im(slot, i, img, h0, w0)

        return img, (h0, w0), (h, w)

    def _generate_self_modal(self, rgb_img):
        """
        生成自体模态图像（当X模态图像缺失时）

        Args:
            rgb_img (np.ndarray): RGB图像

        Returns:
            np.ndarray: 生成的X模态图像
        """
        # 简单的自体模态生成策略：
        # 1. 转换为灰度图
        # 2. 应用边缘检测
        # 3. 转换回3通道

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊和边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 转换为3通道
        x_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return x_img

    def cache_images(self, cache=None):
        """
        缓存RGB和所有非A模态图像到内存或磁盘

        Args:
            cache (str, bool, or None): 缓存模式
                - True or 'ram': 缓存到内存
                - 'disk': 缓存到磁盘
                - False: 不缓存
                - None: 使用self.cache属性

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> dataset.cache_images('ram')  # 缓存到内存
            >>> dataset.cache_images('disk')  # 缓存到磁盘
            >>> dataset.cache_images()  # 使用self.cache属性
        """
        # Use parameter if provided, otherwise fall back to instance attribute
        cache_mode = cache if cache is not None else self.cache

        if not cache_mode:
            self.x_ims = [None] * self.ni
            self.x_im_hw0 = [None] * self.ni
            self.x_im_hw = [None] * self.ni
            return

        # BaseDataset.__init__ 内调用时契约路径尚未就绪；等精度过滤完成后再建缓存
        if not getattr(self, "_mm_paths_ready", False):
            self._cache_pending = True
            return
        self._cache_pending = False

        # 初始化按槽位缓存骨架（Wave 3.2）
        if self.contract is not None:
            for slot in self.contract.slots[1:]:
                self._mod_ims.setdefault(slot, [None] * self.ni)
                self._mod_im_hw0.setdefault(slot, [None] * self.ni)
                self._mod_im_hw.setdefault(slot, [None] * self.ni)
                self._mod_buffers.setdefault(slot, [])

        # 传统缓存模式
        # 初始化多模态缓存存储
        self.x_ims = [None] * self.ni
        self.x_im_hw0 = [None] * self.ni
        self.x_im_hw = [None] * self.ni

        # 调用父类方法缓存RGB图像
        super().cache_images(cache_mode)

        # 按（非 A）槽位缓存其余模态（Wave 3.2）
        if self.contract is not None:
            self._cache_mod_images(cache_mode)

    def _cache_x_modality_images(self, cache):
        """缓存X模态图像"""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self._cache_x_images_to_disk if cache == "disk" else self._load_x_image_for_cache

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    x_npy_file = Path(x) if x is not None else self._get_x_npy_file(i)
                    if x_npy_file.exists():
                        b += x_npy_file.stat().st_size
                else:  # 'ram'
                    if x is not None:
                        self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i] = x
                        b += self.x_ims[i].nbytes if self.x_ims[i] is not None else 0
                pbar.desc = f"{self.prefix}Caching {self.x_modality} images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def _load_x_image_for_cache(self, i):
        """为缓存加载X模态图像（使用预计算的X路径）"""
        try:
            if i < len(self._x_paths) and self._x_paths[i] is not None:
                x_path = self._x_paths[i]
            elif self.im_files and i < len(self.im_files):
                x_path = self._find_corresponding_x_image(self.im_files[i])
            else:
                x_path = None
            if x_path is None:
                raise FileNotFoundError(f"X模态路径未预计算: index={i}")

            # 加载X模态图像
            x_img = self._load_x_modality(x_path)

            # 获取RGB图像信息用于尺寸匹配
            if self.ims[i] is not None:
                # 如果RGB已缓存，使用其尺寸
                rgb_img = self.ims[i]
                h0, w0 = self.im_hw0[i]
                target_shape = rgb_img.shape[:2]
            else:
                # 如果RGB未缓存，先加载RGB获取目标尺寸
                rgb_img, (h0, w0), target_shape = self.load_image(i)

            # 调整X模态图像尺寸以匹配RGB
            if x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))

            return x_img, (h0, w0), x_img.shape[:2]

        except FileNotFoundError as e:
            LOGGER.warning(f"无法加载X模态图像 {i}: {e}")
            return None, None, None

    def _cache_x_images_to_disk(self, i):
        """Cache one X-modality image into the isolated multimodal cache directory."""
        x_npy_file = self._get_x_npy_file(i)
        if x_npy_file.exists():
            return x_npy_file

        try:
            result = self._load_x_image_for_cache(i)
            if result[0] is not None:
                x_img, _, _ = result
                x_npy_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(x_npy_file.as_posix(), x_img, allow_pickle=False)
        except Exception as e:
            LOGGER.warning(f"无法缓存X模态图像到磁盘 {i}: {e}")
        return x_npy_file

    def _dataset_root_from_primary_path(self, primary_path: Union[str, Path]) -> Path:
        """Resolve dataset root from a primary modality sample path."""
        primary_path = Path(primary_path)
        primary_dir = getattr(self, "primary_modality_dir", None) or "images"
        parts = primary_path.parts

        try:
            idx = len(parts) - 1 - parts[::-1].index(primary_dir)
            root_parts = parts[:idx]
            if root_parts:
                return Path(*root_parts)
        except ValueError:
            pass

        try:
            return primary_path.parent.parent.parent
        except IndexError:
            return primary_path.parent

    def _primary_relative_path_for_cache(self, primary_path: Union[str, Path]) -> Path:
        """Return the path under the primary modality directory, preserving split/subdirs."""
        primary_path = Path(primary_path)
        primary_dir = getattr(self, "primary_modality_dir", None) or "images"
        parts = primary_path.parts

        try:
            idx = len(parts) - 1 - parts[::-1].index(primary_dir)
            rel_parts = parts[idx + 1 :]
            if rel_parts:
                return Path(*rel_parts)
        except ValueError:
            pass

        if primary_path.parent.name:
            return Path(primary_path.parent.name) / primary_path.name
        return Path(primary_path.name)

    def _get_x_cache_root(self, primary_path: Union[str, Path]) -> Path:
        """Return isolated X-modality cache root outside primary image scan dirs."""
        dataset_root = self._dataset_root_from_primary_path(primary_path)
        x_modality = str(getattr(self, "x_modality", "x") or "x")
        return dataset_root / self.mm_cache_root_name / x_modality

    def _refresh_x_npy_files(self) -> None:
        """Refresh per-sample X cache file paths after im_files changes."""
        self.x_npy_files = [self._get_x_npy_file(i) for i in range(getattr(self, "ni", 0))]

    def _get_x_npy_file(self, i):
        """Return isolated X-modality .npy cache path for sample index i."""
        primary_path = Path(self.im_files[i])
        relative_path = self._primary_relative_path_for_cache(primary_path).with_suffix(".npy")
        return self._get_x_cache_root(primary_path) / relative_path

    # ── 按槽位缓存（Wave 3.2：泛化 _cache_x_im / _get_x_npy_file）──────────

    def _get_mod_cache_root(self, slot: str, primary_path) -> Path:
        """mm_cache/<modality_name>/ 目录（Wave 3.2）。"""
        name = self.contract.modality_names[self.contract.slots.index(slot)]
        return self._dataset_root_from_primary_path(primary_path) / self.mm_cache_root_name / name

    def _get_mod_npy_file(self, slot: str, i: int):
        """按槽位的 .npy 磁盘缓存路径。"""
        if not getattr(self, 'im_files', None) or i >= len(self.im_files):
            return None
        primary_path = Path(self.im_files[i])
        relative_path = self._primary_relative_path_for_cache(primary_path).with_suffix(".npy")
        return self._get_mod_cache_root(slot, primary_path) / relative_path

    def _cache_mod_im(self, slot: str, i: int, img, h0, w0):
        """按槽位写回内存缓存 + LRU 驱逐（与 RGB buffer 模式对齐）。

        RAM 缓存的唯一写者是 cache_images/_cache_mod_images（父进程构建、worker COW 共享）；
        worker 内禁止写回，否则每个 worker 会各自复制全量模态数据。
        """
        if self.cache == "ram" or not self.augment:
            return
        if slot not in self._mod_ims or self._mod_ims[slot] is None or len(self._mod_ims[slot]) != self.ni:
            self._mod_ims[slot] = [None] * self.ni
            self._mod_im_hw0[slot] = [None] * self.ni
            self._mod_im_hw[slot] = [None] * self.ni
            self._mod_buffers[slot] = []
        self._mod_ims[slot][i] = img
        self._mod_im_hw0[slot][i] = (h0, w0)
        self._mod_im_hw[slot][i] = img.shape[:2]
        buf = self._mod_buffers[slot]
        buf.append(i)
        if 1 < len(buf) >= self.max_buffer_length:
            j = buf.pop(0)
            self._mod_ims[slot][j] = None
            self._mod_im_hw0[slot][j] = None
            self._mod_im_hw[slot][j] = None

    def _load_mod_image_for_cache(self, slot: str, i: int, target_shape=None):
        """为缓存加载指定槽位图像；target_shape 与训练期 RGB 尺寸对齐。"""
        name = self.contract.modality_names[self.contract.slots.index(slot)]
        try:
            return self._load_modality_cached(i, name, target_shape=target_shape)
        except Exception as e:
            LOGGER.warning(f"无法加载模态 {name} 图像 {i}: {e}")
            return None, None, None

    def _cache_mod_images_to_disk(self, slot: str, i: int):
        """按槽位写磁盘缓存。"""
        npy_file = self._get_mod_npy_file(slot, i)
        if npy_file is None or npy_file.exists():
            return npy_file
        try:
            result = self._load_mod_image_for_cache(slot, i)
            if result[0] is not None:
                img = result[0]
                npy_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy_file.as_posix(), img, allow_pickle=False)
        except Exception as e:
            LOGGER.warning(f"无法缓存模态图像到磁盘 slot={slot} i={i}: {e}")
        return npy_file

    def _cache_mod_images(self, cache):
        """按槽位缓存全部非 A 模态图像（Wave 3.2，替代 _cache_x_modality_images）。"""
        gb = 1 << 30
        for slot in self.contract.slots[1:]:
            name = self.contract.modality_names[self.contract.slots.index(slot)]
            b = 0
            if cache == "disk":
                fcn = lambda idx, s=slot: self._cache_mod_images_to_disk(s, idx)
            else:
                # RAM 缓存尺寸必须与训练期 RGB resize 结果一致，避免 worker 命中时二次 resize
                fcn = lambda idx, s=slot: self._load_mod_image_for_cache(s, idx, target_shape=self.im_hw[idx])
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(self.ni))
                pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, x in pbar:
                    if cache == "disk":
                        f = Path(x) if x is not None else self._get_mod_npy_file(slot, i)
                        if f is not None and f.exists():
                            b += f.stat().st_size
                    else:
                        if x is not None and x[0] is not None:
                            self._mod_ims.setdefault(slot, [None] * self.ni)
                            self._mod_im_hw0.setdefault(slot, [None] * self.ni)
                            self._mod_im_hw.setdefault(slot, [None] * self.ni)
                            self._mod_ims[slot][i], self._mod_im_hw0[slot][i], self._mod_im_hw[slot][i] = x
                            b += x[0].nbytes
                    pbar.desc = f"{self.prefix}Caching {name} images ({b / gb:.1f}GB {cache})"
                pbar.close()

    def load_image(self, i, rect_mode=True):
        """重写load_image方法，支持多模态缓存加载"""
        # 加载RGB图像（使用父类方法）
        rgb_img, ori_shape, resized_shape = super().load_image(i, rect_mode)
        return rgb_img, ori_shape, resized_shape

    def _load_x_image_cached(self, i, target_shape=None):
        """从缓存或磁盘加载X模态图像"""
        # 检查内存缓存
        if self.x_ims is not None and self.x_ims[i] is not None:
            return self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i]

        # 检查磁盘缓存
        x_npy_file = self._get_x_npy_file(i)
        if x_npy_file.exists():
            try:
                x_img = np.load(x_npy_file, allow_pickle=False)
                h0, w0 = x_img.shape[:2]
                if target_shape and x_img.shape[:2] != target_shape:
                    x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))
                self._cache_x_im(i, x_img, h0, w0)
                return x_img, (h0, w0), x_img.shape[:2]
            except Exception as e:
                LOGGER.warning(f"加载X模态缓存文件失败 {x_npy_file}: {e}")

        # 从原始文件加载（使用预计算的X路径）
        try:
            if i < len(self._x_paths) and self._x_paths[i] is not None:
                x_path = self._x_paths[i]
            elif self.im_files and i < len(self.im_files):
                x_path = self._find_corresponding_x_image(self.im_files[i])
            else:
                x_path = None
            if x_path is None:
                raise FileNotFoundError(f"X模态路径未预计算: index={i}")
            x_img = self._load_x_modality(x_path)

            h0, w0 = x_img.shape[:2]  # 原始尺寸，resize前捕获

            # 如果提供了目标尺寸，调整X模态图像尺寸
            if target_shape and x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))

            self._cache_x_im(i, x_img, h0, w0)
            return x_img, (h0, w0), x_img.shape[:2]

        except FileNotFoundError as e:
            LOGGER.error(f"无法加载X模态图像 {i}: {e}")
            raise

    def _cache_x_im(self, i, x_img, h0, w0):
        """X模态写回缓存 + LRU驱逐，与RGB的buffer/ims缓存模式对齐。

        RAM 缓存由 cache_images 在父进程一次性构建；worker 内禁止写回。

        Args:
            i: 样本索引
            x_img: 已resize到目标尺寸的X模态图像
            h0, w0: resize前的原始尺寸
        """
        if self.cache == "ram" or not self.augment:
            return
        if self.x_ims is None:
            self.x_ims = [None] * self.ni
            self.x_im_hw0 = [None] * self.ni
            self.x_im_hw = [None] * self.ni
        self.x_ims[i] = x_img
        self.x_im_hw0[i] = (h0, w0)
        self.x_im_hw[i] = x_img.shape[:2]
        self._x_buffer.append(i)
        if 1 < len(self._x_buffer) >= self.max_buffer_length:
            j = self._x_buffer.pop(0)
            self.x_ims[j] = None
            self.x_im_hw0[j] = None
            self.x_im_hw[j] = None

    def check_cache_ram(self, safety_margin=0.5):
        """按契约全部槽位估算 RAM 缓存需求，并给出明确的预算判定。

        cache=True 是训练的既定路径：本方法不做静默降级，只负责把需求算准；
        预算不足时 fail-fast，避免进入必然触发内核 OOM 的状态。
        """
        gb = 1 << 30
        contract = self.contract
        if contract is None or not getattr(self, "im_files", None):
            return True

        n = min(self.ni, 30)
        sample_indices = random.sample(range(self.ni), n)
        details = []
        total_bytes = 0
        for slot, name in zip(contract.slots, contract.modality_names):
            slot_bytes = 0
            for i in sample_indices:
                if slot == contract.slots[0]:
                    path = self.im_files[i]
                else:
                    path = self._find_corresponding_x_image(self.im_files[i], modality_name=name)
                try:
                    if slot == contract.slots[0] or Path(path).suffix.lower() in (".npy", ".npz", ".tif", ".tiff"):
                        img = self._read_cache_source(path)
                    else:
                        # 非 A 模态按原通道读取，避免 cv2_flag=IMREAD_COLOR 把单通道深度图放大 3 倍
                        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if img is not None and img.ndim == 2:
                            img = img[..., None]
                except Exception:
                    img = None
                if img is None:
                    continue
                ratio = self.imgsz / max(img.shape[:2])
                slot_bytes += img.nbytes * ratio**2
            projected = slot_bytes / n * self.ni
            details.append((name, projected))
            total_bytes += projected

        mem_required = total_bytes * (1 + safety_margin)
        mem = psutil.virtual_memory()
        budget = mem.available * 0.7
        detail_str = " + ".join(f"{name}={size / gb:.1f}GB" for name, size in details)
        LOGGER.info(
            f"{self.prefix}多模态 RAM 缓存: {detail_str} = {total_bytes / gb:.1f}GB "
            f"(含 {int(safety_margin * 100)}% 余量 {mem_required / gb:.1f}GB; "
            f"预算 {budget / gb:.1f}/{mem.available / gb:.1f}GB 可用)"
        )
        if mem_required > budget:
            raise RuntimeError(
                f"{self.prefix}RAM 缓存需要 {mem_required / gb:.1f}GB，"
                f"超出预算 {budget / gb:.1f}GB（可用 {mem.available / gb:.1f}GB）。"
                f"明细: {detail_str}。请减少 workers/batch，或显式改用 cache='disk'。"
            )
        self._ram_budget_checked = True
        return True

    def check_cache_disk(self, safety_margin=0.5):
        """Check if there is enough disk space for both primary and X-modality disk caches."""
        import shutil

        primary_cache_ok = super().check_cache_disk(safety_margin)
        if not primary_cache_ok:
            return False

        if not getattr(self, "im_files", None):
            return primary_cache_ok

        cache_root = self._get_x_cache_root(self.im_files[0])
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.cache = None
            LOGGER.warning(f"{self.prefix}Skipping X modality disk cache, cache root not writeable: {cache_root} ({e})")
            return False

        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        samples = 0
        for _ in range(n):
            try:
                rgb_path = random.choice(self.im_files)
                x_path = self._find_corresponding_x_image(rgb_path)
                x_img = self._load_x_modality(x_path)
                b += x_img.nbytes
                samples += 1
            except Exception:
                continue

        if samples == 0:
            return primary_cache_ok

        disk_required = b * self.ni / samples * (1 + safety_margin)
        total, used, free = shutil.disk_usage(cache_root)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required to cache {self.x_modality} images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching {self.x_modality} images"
            )
            return False
        return True


class GroundingDataset(YOLODataset):
    """
    Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            max_samples (int): Maximum number of samples to load for text augmentation.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> List:
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: List[Dict[str, Any]]) -> None:
        """
        Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided
        labels matches the expected count for known datasets. It performs validation
        against a predefined set of datasets with known instance counts.

        Args:
            labels (List[Dict[str, Any]]): List of label dictionaries, where each dictionary
                contains dataset annotations. Each label dict must have a 'bboxes' key with
                a numpy array or tensor containing bounding box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count
                for a recognized dataset.

        Note:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (Dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Dataset class for image classification tasks extending torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return subset of data and targets corresponding to given indices.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """
        Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> Dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> List[Tuple]:
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples


class YOLOMultiModalClassifyDataset(MultiModalImageIOMixin):
    """
    多模态分类数据集（RGB+X）

    设计目标：在 `ultralytics/data/dataset.py` 内与现有数据集实现保持一致的组织方式，
    并沿用 YOLO 系列统一的数据管理范式（YAML + images/labels），而不是 torchvision ImageFolder 目录分类范式。

    数据组织结构（示例）:
        dataset/
        ├── images/
        │   ├── train/
        │   │   └── img001.jpg
        │   └── val/
        ├── images_depth/          # X模态
        │   ├── train/
        │   │   └── img001.png     # 同名，可不同扩展
        │   └── val/
        └── labels/
            ├── train/
            │   └── img001.txt     # 第一行第一列: cls_id
            └── val/

    说明：
        - 标签文件仅解析第一行第一列为 cls_id。
        - RGB 的 torchvision 分类增强仅作用在 RGB 上；X 模态使用对齐 + resize 与 RGB 输出尺寸一致后再拼接。
        - 本类对标签缺失/越界、X 模态缺失等情况采取严格报错（Fail-Fast），避免静默跳过样本。
    """

    def __init__(
        self,
        img_path: str,
        data: Dict[str, Any],
        args,
        augment: bool = False,
        prefix: str = "",
        precision_report_dir=None,
        **kwargs,
    ):
        self.data = data
        self.args = args
        self.augment = augment
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.precision_report_dir = precision_report_dir
        self.precision_contract = {"rgb": "legacy", "x": "legacy"}
        self.precision_report = None
        self.precision_report_path = None

        # 解析多模态配置
        self._parse_multimodal_config()

        # Dataset fraction (only for training/augment)
        self.fraction = float(getattr(args, "fraction", 1.0)) if augment else 1.0

        # 获取图像文件列表（复用 BaseDataset.get_img_files 的统一扫描/校验逻辑）
        from ultralytics.data.utils import MM_PRIMARY_FORMATS
        self.im_files = BaseDataset.get_img_files(self, img_path, allowed_formats=MM_PRIMARY_FORMATS)

        # 解析标签 & 预检查 X 模态文件存在性
        self.labels = self._get_labels_strict()
        self.samples = self._build_samples()

        # 缓存相关（对齐 ClassificationDataset 行为）
        self.cache_ram = getattr(args, "cache", False) is True or str(getattr(args, "cache", "")).lower() == "ram"
        if self.cache_ram:
            LOGGER.warning(
                "MultiModal Classification `cache_ram` has known memory leak, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(getattr(args, "cache", "")).lower() == "disk"

        # 使用 LetterBox 预处理（与检测任务统一）
        self.imgsz = getattr(args, "imgsz", 224)
        from ultralytics.data.augment import LetterBox
        self.letterbox = LetterBox(
            new_shape=self.imgsz,
            auto=False,
            scale_fill=False,
            scaleup=augment,  # 训练时允许放大，验证时不放大
            center=True,
            stride=32
        )

        # dataset-run precision contract: 采样探测 -> 过滤 -> 写报告
        self._initialize_classify_precision_contract()

        LOGGER.info(
            f"{self.prefix}MultiModal Classification Dataset initialized: "
            f"samples={len(self.samples)}, modalities=[{getattr(self, 'primary_modality', 'rgb')},{self.x_modality}], "
            f"Pch={getattr(self, 'primary_ch', 3)}, Xch={self.expected_xch}"
        )

    def _parse_multimodal_config(self) -> None:
        """解析多模态配置（N 模态支持版本）"""
        from ultralytics.utils import LOGGER
        from ultralytics.nn.mm.utils import get_modality_path_from_data
        
        # 优先 modality_used（新默认），兼容旧 models 字段
        if "modality_used" in self.data:
            modality_used = self.data["modality_used"]
        elif "models" in self.data:
            modality_used = self.data["models"]
        else:
            raise ValueError(
                f"{self.prefix}缺少必需字段 `modality_used`（示例: ['rgb','depth'] 或 ['thermal','depth']）。"
            )
        
        # N 模态支持:至少 2 个模态
        if not isinstance(modality_used, list) or len(modality_used) < 2:
            raise ValueError(f"{self.prefix}modality_used 必须是包含至少 2 个模态的列表，当前: {modality_used}")
        
        # 向后兼容:保持 primary_modality/secondary_modality 属性
        self.primary_modality = modality_used[0]
        self.secondary_modality = modality_used[1]
        self.x_modality = self.secondary_modality  # backward compat
        
        # 解析所有模态的目录
        self.primary_modality_dir = get_modality_path_from_data(self.data, self.primary_modality)
        self.secondary_modality_dir = get_modality_path_from_data(self.data, self.secondary_modality)
        self.x_modality_dir = self.secondary_modality_dir  # backward compat
        self.x_modality_suffix = self.data.get("x_modality_suffix", None)
        
        # 存储完整模态列表以供 N 模态加载器使用
        self.modality_list = modality_used
        self.modality_dirs = {mod: get_modality_path_from_data(self.data, mod) for mod in modality_used}
        
        # 日志输出
        if len(modality_used) > 2:
            LOGGER.info(f"MultiModal: {len(modality_used)}-模态配置: {'+'.join(modality_used)}")
        
        from ultralytics.nn.mm.utils import resolve_multimodal_channel_contract
        _dc = resolve_multimodal_channel_contract(data=self.data)
        self.expected_xch = _dc.secondary_ch  # deprecated N==2 显示
        self.secondary_ch = _dc.secondary_ch
        self.nc = int(self.data.get("nc", len(self.data.get("names", {}))))
        self.names = self.data.get("names", {})

        # Auto-detect channel counts (对双模态兼容)
        from pathlib import Path
        from ultralytics.data.dataset import auto_detect_modality_channels
        
        dataset_root = Path(self.data.get('path', '.'))
        self.primary_ch = self.data.get('Pch', None)
        if self.primary_ch is None:
            self.primary_ch = auto_detect_modality_channels(dataset_root, self.primary_modality_dir, split="train")
        if self.secondary_ch is None:
            self.secondary_ch = auto_detect_modality_channels(dataset_root, self.secondary_modality_dir, split="train")

    def _img2label_path(self, img_path: Union[str, Path]) -> Path:
        """将图像路径转换为标签路径: .../<images>/<split>/img.jpg -> .../labels/<split>/img.txt"""
        img_path = Path(img_path)
        parts = list(img_path.parts)
        for i, part in enumerate(parts):
            if part == getattr(self, 'primary_modality_dir', None) or (
                getattr(self, 'primary_modality_dir', None) is None and part == "images"
            ):
                parts[i] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def _parse_cls_label(self, label_path: Path) -> int:
        """解析分类标签文件：只读取第一行第一列作为 cls_id"""
        if not label_path.exists():
            raise FileNotFoundError(f"{self.prefix}标签文件不存在: {label_path}")

        with label_path.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                tok = s.split()
                cls_id = int(tok[0])
                if not (0 <= cls_id < self.nc):
                    raise ValueError(f"{self.prefix}cls_id 越界: {cls_id}, nc={self.nc}, file={label_path}")
                return cls_id

        raise ValueError(f"{self.prefix}标签文件为空或无有效行: {label_path}")

    def _get_labels_strict(self) -> List[int]:
        """扫描并解析全部标签；任何错误将汇总后抛出。"""
        labels: List[int] = []
        errors: List[str] = []

        pbar = TQDM(self.im_files, desc=f"{self.prefix}Scanning labels...", disable=LOCAL_RANK not in {-1, 0})
        for im_file in pbar:
            label_path = self._img2label_path(im_file)
            try:
                labels.append(self._parse_cls_label(label_path))
            except Exception as e:
                errors.append(f"{label_path}: {e}")
                labels.append(-1)

        if errors:
            preview = "\n".join(errors[:20])
            more = f"\n... 另外 {len(errors) - 20} 个错误" if len(errors) > 20 else ""
            raise ValueError(f"{self.prefix}发现无效标签文件，共 {len(errors)} 个：\n{preview}{more}")

        return labels

    def _build_samples(self) -> List[List[Any]]:
        """构建样本列表并预检查 X 模态路径存在性。"""
        samples: List[List[Any]] = []
        errors: List[str] = []

        for im_file, cls_id in zip(self.im_files, self.labels):
            if cls_id < 0:
                # labels 已严格校验，这里仅作防御
                continue

            x_path = self.find_corresponding_x_image(
                rgb_path=im_file,
                x_modality_dir=self.x_modality_dir,
                x_modality_suffix=self.x_modality_suffix,
            )
            if not Path(x_path).exists():
                errors.append(f"X模态缺失: rgb={im_file} -> x={x_path}")

            npy_path = Path(im_file).with_suffix(".npy")
            samples.append([im_file, cls_id, npy_path, None, x_path])

        if errors:
            preview = "\n".join(errors[:20])
            more = f"\n... 另外 {len(errors) - 20} 个错误" if len(errors) > 20 else ""
            raise FileNotFoundError(f"{self.prefix}发现 X 模态缺失，共 {len(errors)} 个：\n{preview}{more}")

        return samples

    def _initialize_classify_precision_contract(self) -> None:
        """Resolve dataset-run precision contract for classify dataset.

        Mirrors YOLOMultiModalImageDataset._initialize_precision_contract but
        adapts to classify's data structure (self.samples instead of self.labels).
        """
        from ultralytics.data.multimodal.precision_contract import (
            filter_dataset_by_contract,
            log_dataset_precision_summary,
            probe_dataset_precision,
        )

        if not getattr(self, "im_files", None):
            return

        report = probe_dataset_precision(self)
        keep_indices, filtered = filter_dataset_by_contract(self, report["contracts"])
        split = Path(self.im_files[0]).parent.name if self.im_files else "unknown"
        report["split"] = split
        report["filtered_count"] = len(filtered)
        report["filtered_examples"] = filtered

        if keep_indices and len(keep_indices) < len(self.im_files):
            # Filter samples / im_files / labels to keep only matching indices
            self.samples = [self.samples[i] for i in keep_indices]
            self.im_files = [self.im_files[i] for i in keep_indices]
            self.labels = [self.labels[i] for i in keep_indices]
            report["filter_applied"] = True
        elif not keep_indices and filtered:
            LOGGER.warning(
                "MultiModal precision contract would filter all %d classify samples for split=%s, keeping original dataset",
                len(self.im_files),
                split,
            )
            report["filter_applied"] = False
            report["filtered_count"] = 0
            report["filtered_examples"] = []
        else:
            report["filter_applied"] = False

        self.precision_contract = dict(report["contracts"])
        self.precision_report = report
        self.precision_report_path = None
        log_dataset_precision_summary(report, None)

    def _find_corresponding_x_image(self, rgb_path):
        """Bridge: precision_contract._resolve_x_path calls dataset._find_corresponding_x_image."""
        return self.find_corresponding_x_image(
            rgb_path=rgb_path,
            x_modality_dir=self.x_modality_dir,
            x_modality_suffix=self.x_modality_suffix,
        )

    def build_transforms(self):
        """LetterBox 已在 __init__ 中初始化，此方法仅为兼容性保留"""
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        im_file, cls_id, npy_path, cached_img, x_path = self.samples[index]

        # 加载 RGB 图像（BGR）
        if self.cache_ram and cached_img is not None:
            rgb_img = cached_img
        elif self.cache_disk:
            if not npy_path.exists():
                rgb_img = cv2.imread(im_file)
                if rgb_img is None:
                    raise ValueError(f"{self.prefix}无法读取RGB图像: {im_file}")
                np.save(npy_path.as_posix(), rgb_img, allow_pickle=False)
            rgb_img = np.load(npy_path)
        else:
            rgb_img = cv2.imread(im_file)

        if rgb_img is None:
            raise ValueError(f"{self.prefix}无法读取RGB图像: {im_file}")

        # 加载并对齐 X 模态图像（仍在原始分辨率空间）
        x_img = self.load_x_modality(x_path)
        x_img = self.align_x_to_rgb(x_img, rgb_img.shape[:2])
        x_img = self.validate_x_channels(x_img, self.expected_xch, strict=False, x_path_hint=str(x_path))

        # Mixin 在 expected_xch>3 时会 warning 而不是 fail-fast，这里强制校验
        final_xch = 1 if len(x_img.shape) == 2 else (x_img.shape[2] if len(x_img.shape) == 3 else -1)
        if final_xch != self.expected_xch:
            raise ValueError(
                f"{self.prefix}X模态通道数不匹配: expected={self.expected_xch}, got={final_xch}, file={x_path}"
            )

        # 确保 X 是 3D
        if len(x_img.shape) == 2:
            x_img = x_img[:, :, np.newaxis]

        # 合并为多通道图像，统一使用 LetterBox
        primary_img = rgb_img  # primary modality image
        secondary_img = x_img  # secondary modality image
        merged = np.concatenate([primary_img, secondary_img], axis=2)  # [H, W, Pch+Xch]
        merged_lb = self.letterbox(image=merged)  # [H', W', Pch+Xch]

        # 分离主模态和附模态
        primary_ch_count = getattr(self, 'primary_ch', 3) or 3
        primary_lb = merged_lb[:, :, :primary_ch_count]
        secondary_lb = merged_lb[:, :, primary_ch_count:]

        # Precision contract: use dataset-run contract resolved at __init__ time
        from ultralytics.data.multimodal.utils import to_tensor_rgb, to_tensor_x

        precision_contract = self.precision_contract

        # Tensorize: 根据 primary_modality 选择正确的 tensorize helper
        if is_rgb_modality(self.primary_modality):
            # RGB 模态: BGR->RGB + HWC->CHW + contract-aware normalize
            primary_rgb = primary_lb[:, :, ::-1].copy()
            primary_tensor = to_tensor_rgb(primary_rgb, precision=precision_contract["rgb"])
        else:
            # 非 RGB 主模态: HWC->CHW + contract-aware normalize
            primary_tensor = to_tensor_x(primary_lb, precision=precision_contract["rgb"])

        secondary_tensor = to_tensor_x(secondary_lb, precision=precision_contract["x"])

        return {
            "img": torch.cat([primary_tensor, secondary_tensor], dim=0),
            "cls": cls_id,
            "precision_contract": precision_contract,
        }


# ===== SemSeg Format + Dataset =====

class SemSegFormat:
    """Convert semantic mask to torch.long after standard Format has processed img/instances."""

    def __call__(self, labels: dict) -> dict:
        sem_mask = labels.get("sem_mask", None)
        if sem_mask is None:
            raise KeyError("SemSeg sample missing 'sem_mask'")
        if isinstance(sem_mask, np.ndarray):
            labels["sem_mask"] = torch.from_numpy(np.ascontiguousarray(sem_mask)).long()
        elif isinstance(sem_mask, torch.Tensor):
            labels["sem_mask"] = sem_mask.long()
        else:
            raise TypeError(f"sem_mask must be ndarray or Tensor, got {type(sem_mask).__name__}")
        return labels


class YOLOMultiModalSemSegDataset(YOLOMultiModalImageDataset):
    """YOLOMM pure semantic segmentation dataset using image mask labels."""

    SEM_LABEL_EXTS = (".png", ".jpg", ".jpeg")
    _warned_jpg_label = False
    _warned_multichannel_label = False
    _warned_invalid_values = False

    @classmethod
    def _warn_once(cls, attr_name: str, message: str):
        if not getattr(cls, attr_name, False):
            setattr(cls, attr_name, True)
            LOGGER.warning(message)

    def _parse_label_mapping(self, mapping):
        """Normalize label_mapping from dataset YAML into int->int ids.

        Args:
            mapping: Raw label_mapping value from dataset YAML (dict or None).

        Returns:
            Normalized dict mapping source pixel value (int) -> target class ID (int).
            Empty dict if mapping is None.
        """
        if mapping is None:
            return {}
        if not isinstance(mapping, dict):
            raise TypeError(f"Expected 'label_mapping' to be a dict, got {type(mapping).__name__}")
        normalized = {}
        for src, dst in mapping.items():
            src = int(src)
            if dst is None:
                dst = 255
            elif isinstance(dst, str):
                dst = dst.strip()
                dst = 255 if dst == "ignore_label" else int(dst)
            else:
                dst = int(dst)
            normalized[src] = dst
        return normalized

    def convert_label(self, label, inverse=False):
        """Convert label values using label_mapping.

        Args:
            label: np.ndarray, segmentation mask.
            inverse: If True, map target->source (for saving predictions).
        """
        if not self.label_mapping:
            return label
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def _sanitize_invalid_sem_values(self, mask: np.ndarray, nc: int, ignore_index: int) -> np.ndarray:
        """Map invalid class IDs to ignore_index. Returns sanitized array."""
        nc_effective = 2 if nc == 1 else nc  # nc=1 binary: valid values {0,1}
        invalid = (mask != ignore_index) & ((mask < 0) | (mask >= nc_effective))
        if invalid.any():
            self._warn_once("_warned_invalid_values",
                            f"SemSeg mask contains {invalid.sum()} invalid class values, mapped to ignore_index={ignore_index}")
            mask = mask.copy()
            mask[invalid] = ignore_index
        return mask

    def _load_sem_mask(self, sem_file: str, rgb_shape: tuple) -> np.ndarray:
        """Load semantic mask: grayscale, resize to img, sanitize, apply label_mapping.

        Aligns with upstream SemanticDataset.load_mask():
        - IMREAD_GRAYSCALE forces single-channel
        - Auto-resize to img dimensions (tolerant, not error)
        - nc=1 handles 1-bit PNG
        - Applies label_mapping then sanitize
        """
        if not sem_file:
            raise FileNotFoundError("sem_file is None or empty")
        mask = cv2.imread(sem_file, cv2.IMREAD_GRAYSCALE)  # upstream: force grayscale
        if mask is None:
            raise FileNotFoundError(f"Cannot read SemSeg mask: {sem_file}")

        ext = os.path.splitext(sem_file)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            self._warn_once("_warned_jpg_label",
                            f"SemSeg label in JPEG format ({sem_file}). PNG is recommended.")

        # nc=1 binary: handle 1-bit PNG (value range {0, 255} -> {0, 1})
        nc = int(self.data.get("nc", 80))
        if nc == 1:
            from PIL import Image as PILImage
            with PILImage.open(sem_file) as im:
                if im.mode == "1":  # true 1-bit PNG
                    mask[mask == 255] = 1

        # Resize to img dimensions (upstream: tolerant, not error)
        h, w = rgb_shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # label_mapping conversion
        if self.label_mapping:
            mask = self.convert_label(mask, inverse=False)

        # sanitize invalid values
        ignore_index = int(self.data.get("ignore_index", getattr(self, "ignore_index", 255)))
        mask = self._sanitize_invalid_sem_values(mask, nc=nc, ignore_index=ignore_index)
        return mask.astype(np.uint8)  # uint8, not int64

    # --- Cache system (Phase 4.4: refactored to use verify_image_mask) ---

    def _semantic_cache_hash(self, sem_files: list) -> str:
        """Cache hash: im_files + sem_files + label_mapping."""
        mapping = json.dumps(self.label_mapping, sort_keys=True, separators=(",", ":"))
        return get_hash(self.im_files + sem_files + [f"label_mapping:{mapping}"])

    def cache_labels(self, path=Path("./labels.cache")):
        """Override: multi-thread verify image-mask pairs."""
        from ultralytics.data.utils import verify_image_mask
        x = {"labels": []}
        nm, nf, nc, msgs = 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_mask,
                iterable=zip(self.im_files, self.sem_files, repeat(self.prefix)),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, mask_file, shape, nm_f, nf_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                nc += nc_f
                if im_file:
                    x["labels"].append({
                        "im_file": im_file,
                        "sem_file": mask_file,
                        "shape": shape,
                        "cls": np.array([], dtype=np.float32),
                        "bboxes": np.zeros((0, 4), dtype=np.float32),
                        "segments": [],
                        "normalized": True,
                        "bbox_format": "xywh",
                    })
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm} missing masks, {nc} corrupt"
            pbar.close()

        x["hash"] = self._semantic_cache_hash(self.sem_files)
        x["results"] = nf, nm, nc, total
        x["msgs"] = msgs
        if x["labels"]:
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        else:
            LOGGER.warning(f"{self.prefix}No valid image-mask pairs found, cache not saved")
        return x

    def get_labels(self):
        """Override: build sem_files via img2label_paths, load cache or rebuild."""
        if not hasattr(self, "label_mapping"):
            self.label_mapping = self._parse_label_mapping(self.data.get("label_mapping"))

        # Force labels/ directory parallel to images/, mask same name with .png suffix
        self.sem_files = img2label_paths(self.im_files, label_dir="labels", suffix=".png")
        cache_path = Path(self.sem_files[0]).parent.with_suffix(".cache")

        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == self._semantic_cache_hash(self.sem_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} masks, {nm} missing, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache.get("msgs"):
                LOGGER.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid image-mask pairs found in {cache_path}. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]
        self.sem_files = [lb["sem_file"] for lb in labels]
        return labels

    def get_image_and_label(self, index: int) -> dict:
        """Return labels with RGB+X multimodal image and sem_mask.

        Overrides parent so that Mosaic donor loading also carries sem_mask.
        """
        label = super().get_image_and_label(index)
        rgb, x_img = self._load_multimodal_components(index)
        label["_rgb"] = rgb
        label["_x"] = x_img
        label["img"] = np.concatenate([rgb, x_img], axis=2)
        label["sem_mask"] = self._load_sem_mask(self.labels[index]["sem_file"],
                                                  rgb_shape=rgb.shape[:2])
        return label

    def __getitem__(self, index):
        label = self.get_image_and_label(index)
        label["precision_contract"] = dict(self.precision_contract)
        label["precision_report_path"] = self.precision_report_path
        label["precision_mode"] = "dataset_run"
        label["index"] = index
        return self.transforms(label)

    def build_transforms(self, hyp=None):
        from ultralytics.data.augment import Compose, Format, LetterBox
        from ultralytics.data.multimodal_augment import mm_semseg_transforms

        ignore = int(getattr(hyp, "ignore_index", getattr(self, "ignore_index", 255)))
        if self.augment:
            transforms = mm_semseg_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, ignore_index=ignore)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                bgr=0.0,
            )
        )
        transforms.append(SemSegFormat())
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate SemSeg samples; stack sem_mask into [B,H,W]."""
        new_batch = YOLODataset.collate_fn(batch)
        sem_mask = new_batch.get("sem_mask")
        if isinstance(sem_mask, (tuple, list)):
            new_batch["sem_mask"] = torch.stack([m.long() for m in sem_mask], 0)
        elif isinstance(sem_mask, torch.Tensor) and sem_mask.ndim == 2:
            new_batch["sem_mask"] = sem_mask.unsqueeze(0).long()
        elif isinstance(sem_mask, torch.Tensor):
            new_batch["sem_mask"] = sem_mask.long()
        else:
            raise KeyError("SemSeg batch missing 'sem_mask'")
        return new_batch


# ===== Depth Format + Dataset =====

class DepthFormat:
    """Convert metric depth GT to float32 tensor [H,W] after Format."""

    def __call__(self, labels: dict) -> dict:
        depth = labels.get("depth", None)
        if depth is None:
            raise KeyError("Depth sample missing 'depth'")
        if isinstance(depth, np.ndarray):
            labels["depth"] = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))
        elif isinstance(depth, torch.Tensor):
            labels["depth"] = depth.float()
        else:
            raise TypeError(f"depth must be ndarray or Tensor, got {type(depth).__name__}")
        if labels["depth"].ndim != 2:
            raise ValueError(f"depth tensor must be [H,W], got {tuple(labels['depth'].shape)}")
        return labels


class YOLOMultiModalDepthDataset(YOLOMultiModalImageDataset):
    """RGB+X dataset with metric depth GT from gt_depth/ (not labels/, not images_depth/)."""

    def __init__(self, *args, **kwargs):
        data = kwargs.get("data") or {}
        self.gt_depth_dir = resolve_depth_gt_dir(data)
        self.depth_scale = float(data.get("depth_scale", 1000))
        super().__init__(*args, **kwargs)
        x_dir_name = Path(str(getattr(self, "x_modality_dir", ""))).name
        if x_dir_name == Path(self.gt_depth_dir).name:
            raise ValueError(
                f"X directory {self.x_modality_dir!r} must not equal GT directory {self.gt_depth_dir!r}"
            )

    def get_labels(self):
        """Do not call YOLODataset.get_labels(): that scans labels/*.txt (detect/semseg)."""
        self.depth_files = img2depth_paths(self.im_files, gt_dir=self.gt_depth_dir)
        missing = [p for p in self.depth_files if not Path(p).is_file()]
        if missing:
            preview = "\n".join(missing[:8])
            raise FileNotFoundError(
                f"{len(missing)} depth GT files missing under {self.gt_depth_dir}/. "
                f"Use metric uint16 PNG or float npy. Examples:\n{preview}"
            )
        labels = []
        for im_file, depth_file in zip(self.im_files, self.depth_files):
            labels.append({
                "im_file": im_file,
                "depth_file": depth_file,
                "cls": np.array([], dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "normalized": True,
                "bbox_format": "xywh",
            })
        return labels

    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)
        rgb, x_img = self._load_multimodal_components(index)
        label["_rgb"] = rgb
        label["_x"] = x_img
        label["img"] = np.concatenate([rgb, x_img], axis=2)
        depth = load_depth(self.labels[index]["depth_file"], depth_scale=self.depth_scale)
        # load_image(rect_mode=True) may already resize RGB; GT must match native RGB, then NEAREST-sync.
        ori_hw = tuple(label.get("ori_shape") or rgb.shape[:2])
        if depth.shape[:2] != ori_hw:
            raise ValueError(
                f"Depth GT shape {depth.shape[:2]} != RGB {ori_hw} for {self.labels[index]['depth_file']}"
            )
        rgb_hw = rgb.shape[:2]
        if depth.shape[:2] != rgb_hw:
            depth = cv2.resize(depth, (rgb_hw[1], rgb_hw[0]), interpolation=cv2.INTER_NEAREST)
        label["depth"] = depth
        return label

    def __getitem__(self, index):
        label = self.get_image_and_label(index)
        label["precision_contract"] = dict(self.precision_contract)
        label["precision_report_path"] = self.precision_report_path
        label["precision_mode"] = "dataset_run"
        label["index"] = index
        return self.transforms(label)

    def build_transforms(self, hyp=None):
        from ultralytics.data.augment import Compose, Format, LetterBox
        from ultralytics.data.multimodal_augment import mm_depth_transforms

        if hyp is not None:
            hyp.mosaic = 0.0
            hyp.mixup = 0.0
            hyp.cutmix = 0.0
            hyp.copy_paste = 0.0
        if self.augment:
            transforms = mm_depth_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, scale_fill=True)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                bgr=0.0,
            )
        )
        transforms.append(DepthFormat())
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = YOLODataset.collate_fn(batch)
        depth = new_batch.get("depth")
        if isinstance(depth, (tuple, list)):
            new_batch["depth"] = torch.stack([d.float() for d in depth], 0).unsqueeze(1)
        elif isinstance(depth, torch.Tensor) and depth.ndim == 2:
            new_batch["depth"] = depth.unsqueeze(0).unsqueeze(0).float()
        elif isinstance(depth, torch.Tensor) and depth.ndim == 3:
            new_batch["depth"] = depth.unsqueeze(1).float()
        elif isinstance(depth, torch.Tensor) and depth.ndim == 4:
            new_batch["depth"] = depth.float()
        else:
            raise KeyError("Depth batch missing 'depth'")
        return new_batch
