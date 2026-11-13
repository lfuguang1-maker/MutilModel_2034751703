# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tracking orchestration for TRACK: sequence iteration -> inference -> tracker -> results."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ultralytics.cfg import get_save_dir
from ultralytics.data.multimodal import SequenceResolver
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import LOGGER, ops

from .saver import TrackSaver


class _BoxDetections:
    """Tracker input adapter for box tasks (must not expose ``xywhr``)."""

    __slots__ = ("conf", "cls", "xywh")

    def __init__(self, conf: np.ndarray, cls: np.ndarray, xywh: np.ndarray):
        self.conf = conf
        self.cls = cls
        self.xywh = xywh


class _OBBDetections:
    """Tracker input adapter for oriented boxes."""

    __slots__ = ("conf", "cls", "xywhr")

    def __init__(self, conf: np.ndarray, cls: np.ndarray, xywhr: np.ndarray):
        self.conf = conf
        self.cls = cls
        self.xywhr = xywhr


def _gmc_image(orig_imgs: Dict, anchor_slot: str) -> Optional[np.ndarray]:
    """Convert the anchor slot image to the 8-bit 3-channel image expected by BOTSORT GMC."""
    img = (orig_imgs or {}).get(anchor_slot)
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo) * 255.0).astype(np.uint8) if hi > lo else np.zeros(img.shape, np.uint8)
    return np.ascontiguousarray(img)


def _build_detections(result, task: str):
    """Build the minimal object expected by ``tracker.update``."""
    if task == "obb":
        obb = result.obb
        return _OBBDetections(
            conf=obb.conf.astype(np.float32),
            cls=obb.cls.astype(np.int32),
            xywhr=obb.xywhr.astype(np.float32),
        )
    boxes = result.boxes
    return _BoxDetections(
        conf=boxes[:, 4].astype(np.float32),
        cls=boxes[:, 5].astype(np.int32),
        xywh=ops.xyxy2xywh(boxes[:, :4]).astype(np.float32),
    )


def _track_result(result, tracker, task: str, classes: Optional[np.ndarray], gmc_img):
    """Filter by class, run tracker, and write ids / selection back to the result."""
    if classes is not None:
        keep = np.isin(result.boxes[:, 5].astype(int), classes)
        if not keep.all():
            result = result.select(np.flatnonzero(keep))
    tracks = tracker.update(_build_detections(result, task), img=gmc_img)
    if len(tracks):
        ids = tracks[:, -4].astype(int)
        idx = tracks[:, -1].astype(int)
    else:
        ids = np.empty(0, dtype=int)
        idx = np.empty(0, dtype=int)
    result.ids = ids
    return result.select(idx)


def run_tracking(
    model,
    sources: Dict[str, Any],
    tracker_cfg,
    stream: bool = False,
    fps: Optional[float] = None,
    strict_match: bool = True,
    classes: Optional[Iterable[int]] = None,
    conf: float = 0.1,
    iou: Optional[float] = None,
    imgsz: Optional[int] = None,
    max_det: Optional[int] = None,
    device: Optional[str] = None,
    save: bool = False,
    save_txt: bool = False,
    save_dir: Optional[Path] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    exist_ok: bool = False,
    line_width: Optional[int] = None,
    font_size: Optional[int] = None,
    show_conf: bool = True,
    show_labels: bool = True,
    trail: int = 30,
    verbose: bool = True,
):
    """Track every frame of every resolved sequence.

    Returns a list when ``stream=False`` and a generator otherwise. Each yielded result
    carries ``ids`` plus ``meta['seq']`` / ``meta['frame']`` / ``meta['fps']``.
    """
    engine = model.build_engine(
        conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, font_size=font_size
    )
    contract = getattr(engine.router, "contract", None)
    if contract is None:
        raise RuntimeError("TRACK 需要带通道契约的多模态引擎")
    sequences = SequenceResolver(verbose=verbose).resolve(sources, contract, strict_match=strict_match, fps=fps)
    class_ids = None if classes is None else np.atleast_1d(np.asarray(classes, dtype=int))
    saver = None
    if save or save_txt:
        if save_dir is None:
            save_dir = get_save_dir(
                SimpleNamespace(task=model.task, mode="track", project=project, name=name, exist_ok=exist_ok)
            )
        saver = TrackSaver(
            save_dir,
            model.task,
            save=save,
            save_txt=save_txt,
            line_width=line_width,
            font_size=font_size,
            show_conf=show_conf,
            show_labels=show_labels,
            trail=trail,
        )
        if verbose:
            LOGGER.info(f"跟踪输出目录: {save_dir}")

    if not stream:
        total_frames = sum(seq.length for seq in sequences)
        if any(seq.kind == "video" for seq in sequences) or total_frames > 300:
            LOGGER.warning(
                "stream=False 会缓存全部帧结果；当前输入含视频或总帧数超过 300，"
                "建议使用 stream=True 以限制内存占用。"
            )

    def _iter_results():
        for seq in sequences:
            if saver:
                saver.start_sequence(seq)
            data_iter = result_iter = None
            completed = False
            detect_ms: List[float] = []
            track_ms: List[float] = []
            count = 0
            try:
                dataset = (
                    engine.build_dataset(seq.to_sample_specs())
                    if seq.kind == "frames"
                    else engine.build_video_dataset(seq)
                )
                tracker = TRACKER_MAP[tracker_cfg.tracker_type](args=tracker_cfg, frame_rate=seq.fps)
                tracker.reset()
                data_iter = iter(dataset)
                result_iter = iter(engine.stream_dataset(data_iter, conf=conf, iou=iou, max_det=max_det))
                while True:
                    t0 = time.perf_counter()
                    try:
                        result = next(result_iter)
                    except StopIteration:
                        break
                    detect_ms.append((time.perf_counter() - t0) * 1000.0)

                    t1 = time.perf_counter()
                    gmc_img = _gmc_image(result.orig_imgs, seq.anchor_slot) if hasattr(tracker, "gmc") else None
                    result = _track_result(result, tracker, model.task, class_ids, gmc_img)
                    result.meta["seq"] = seq.name
                    result.meta.setdefault("frame", count + 1)
                    result.meta["fps"] = seq.fps
                    track_ms.append((time.perf_counter() - t1) * 1000.0)
                    count += 1
                    if saver:
                        saver.write(result, seq)
                    yield result
                completed = True
            finally:
                if result_iter is not None:
                    result_iter.close()
                if data_iter is not None and hasattr(data_iter, "close"):
                    data_iter.close()
                if saver:
                    saver.close_sequence(seq, verify=completed)

            if verbose:
                detect_avg = float(np.mean(detect_ms)) if detect_ms else 0.0
                track_avg = float(np.mean(track_ms)) if track_ms else 0.0
                LOGGER.info(
                    f"序列 {seq.name}: {count} 帧 | 平均检测 {detect_avg:.2f} ms/帧 | "
                    f"平均跟踪 {track_avg:.2f} ms/帧"
                )

    return _iter_results() if stream else list(_iter_results())
