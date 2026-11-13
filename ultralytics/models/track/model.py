# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""TRACK: offline multi-object tracking on trained YOLOMM / RTDETRMM weights."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.models.rtdetrmm.model import RTDETRMM
from ultralytics.models.rtdetrmm.utils import detect_mm
from ultralytics.models.yolo.model import YOLOMM
from ultralytics.utils import LOGGER, YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.patches import torch_load

_SUPPORTED_TASKS = {
    "YOLOMM": {"detect", "segment", "pose", "obb"},
    "RTDETRMM": {"detect", "obb"},
}

_TRACK_DEFAULTS = {
    "tracker": "botsort.yaml",
    "stream": False,
    "fps": None,
    "strict_match": True,
    "classes": None,
    "conf": 0.1,
    "iou": None,
    "imgsz": None,
    "max_det": None,
    "device": None,
    "save": False,
    "save_txt": False,
    "project": None,
    "name": None,
    "exist_ok": False,
    "line_width": None,
    "font_size": None,
    "show_conf": True,
    "show_labels": True,
    "trail": 30,
    "verbose": True,
}

_VAL_DEFAULTS = {
    "split": "val",
    "tracker": "bytetrack.yaml",
    "modality": None,
    "conf": 0.1,
    "iou": None,
    "imgsz": None,
    "max_det": None,
    "device": None,
    "classes": None,
    "save": False,
    "save_txt": False,
    "project": None,
    "name": None,
    "exist_ok": False,
    "verbose": True,
}


class TRACK:
    """Offline tracking wrapper for trained YOLOMM and RTDETRMM checkpoints.

    The class deliberately does not inherit ``Model``. It owns a YOLOMM or RTDETRMM
    detector and borrows that family's multimodal inference engine.
    """

    def __init__(self, model: Union[str, Path], verbose: bool = False):
        path = Path(model)
        if path.suffix.lower() != ".pt":
            raise ValueError(
                f"TRACK 只接受已训练的 .pt 权重，收到 {path}。"
                "支持的组合：YOLOMM detect/segment/pose/obb，RTDETRMM detect/obb。"
            )
        if not path.is_file():
            raise FileNotFoundError(f"TRACK 权重不存在: {path}")

        ckpt = torch_load(str(path), map_location="cpu")
        model_obj = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else None
        if model_obj is None:
            raise ValueError(f"TRACK 无法从权重读取模型对象: {path}")
        family = "RTDETRMM" if "RTDETR" in type(model_obj).__name__ else "YOLOMM"
        task = getattr(model_obj, "task", None) or (ckpt.get("train_args", {}) or {}).get("task")
        if task not in _SUPPORTED_TASKS[family]:
            raise ValueError(
                f"TRACK 不支持 {family} 的 {task!r} 任务。"
                f"支持：YOLOMM detect/segment/pose/obb，RTDETRMM detect/obb。"
            )

        detector_cls = RTDETRMM if family == "RTDETRMM" else YOLOMM
        self.family = family
        self.task = task
        self.verbose = verbose
        self.detector = detector_cls(path, verbose=verbose)

        evidence = detect_mm(path)
        if not evidence.is_multimodal:
            evidence = detect_mm(getattr(self.detector.model, "yaml", None) or {})
        if not evidence.is_multimodal:
            raise ValueError(
                f"TRACK 需要多模态权重，{path} 未通过多模态判据。"
                f"原因：{evidence.reason}；source={evidence.source}。"
                "支持的组合：YOLOMM detect/segment/pose/obb，RTDETRMM detect/obb。"
            )
        if verbose:
            LOGGER.info(f"TRACK 初始化完成: {family} {task}, 权重 {path}")

    @property
    def names(self) -> Dict:
        """Return the detector class-name mapping."""
        return self.detector.names

    @property
    def model(self):
        """Return the underlying torch model."""
        return self.detector.model

    def build_engine(
        self,
        conf: float = 0.1,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        device: Optional[str] = None,
        font_size: Optional[int] = None,
    ):
        """Build the family-specific multimodal inference engine.

        The construction mirrors ``Model.predict``: load the adapter via the detector's
        ``task_map``, call ``setup_model``, then use the adapter's inner engine.
        """
        overrides = dict(self.detector.overrides)
        args = {
            **overrides,
            "conf": conf,
            "iou": iou if iou is not None else overrides.get("iou", 0.7),
            "imgsz": imgsz if imgsz is not None else overrides.get("imgsz", 640),
            "max_det": max_det if max_det is not None else overrides.get("max_det", 300),
            "batch": 1,
            "save": False,
            "mode": "predict",
            "rect": True,
        }
        if device is not None:
            args["device"] = device
        if font_size is not None:
            args["font_size"] = font_size

        adapter_cls = self.detector._smart_load("predictor")
        adapter = adapter_cls(overrides=args, _callbacks=self.detector.callbacks)
        adapter.setup_model(model=self.detector.model, verbose=self.verbose, debug=False)
        return adapter.predictor

    def track(self, sources=None, rgb_source=None, x_source=None, **kwargs):
        """Track all frames of the supplied offline sequence(s)."""
        params = self._validate_kwargs(kwargs, _TRACK_DEFAULTS, "track")
        sources = self._normalize_sources(sources, rgb_source, x_source)
        tracker_cfg = self._load_tracker_cfg(params.pop("tracker"))
        from .runner import run_tracking

        return run_tracking(self, sources=sources, tracker_cfg=tracker_cfg, **params)

    def val(self, data=None, **kwargs):
        """Evaluate tracking on a YOLOMMtrack dataset."""
        params = self._validate_kwargs(kwargs, _VAL_DEFAULTS, "val")
        if data is None:
            raise ValueError("TRACK.val() 需要 data=... 指向 YOLOMMtrack dataset YAML")
        from .val import run_track_val

        return run_track_val(self, data=data, **params)

    @staticmethod
    def _validate_kwargs(kwargs: Dict[str, Any], defaults: Dict[str, Any], mode: str) -> Dict[str, Any]:
        unknown = sorted(set(kwargs) - set(defaults))
        if unknown:
            raise TypeError(
                f"TRACK.{mode}() 不支持参数: {unknown}。"
                f"支持参数: {sorted(defaults)}。"
                "如需实时流、视频抽帧或保存 JSON，请先在计划中扩展对应能力。"
            )
        return {**defaults, **kwargs}

    @staticmethod
    def _normalize_sources(sources, rgb_source, x_source):
        if sources is not None and (rgb_source is not None or x_source is not None):
            raise ValueError("sources 与 rgb_source/x_source 互斥，请只传一种")
        if sources is None:
            if rgb_source is None and x_source is None:
                raise ValueError("需要提供 sources={'A': ...} 或 rgb_source/x_source")
            sources = {"A": rgb_source, "B": x_source}
        return sources

    @staticmethod
    def _load_tracker_cfg(tracker: Union[str, Path]) -> IterableSimpleNamespace:
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
        if cfg.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError(
                f"tracker_type 只支持 bytetrack / botsort，收到 {cfg.tracker_type!r}: {tracker}"
            )
        if getattr(cfg, "with_reid", False):
            raise ValueError(
                "第一版 TRACK 不支持 ReID（with_reid: True）。"
                "请将 tracker YAML 中的 with_reid 改为 False。"
            )
        return cfg
