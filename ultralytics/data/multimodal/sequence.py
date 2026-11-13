# Ultralytics Multimodal Inference - Offline Sequence Data
# Video-pair and frame-directory sequences for multimodal tracking
# Date: 2026-10-17

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ultralytics.data.utils import ARRAY_FORMATS, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER

from .inference_dataset import MultiModalInferenceDataset

SEQUENCE_KIND_VIDEO = "video"
SEQUENCE_KIND_FRAMES = "frames"
_IMAGE_SUFFIXES = {f".{s}" for s in IMG_FORMATS | ARRAY_FORMATS}
_VIDEO_SUFFIXES = {f".{s}" for s in VID_FORMATS}


def natural_sort_key(value: Union[str, Path]) -> Tuple:
    """Build a natural sort key so ``2`` sorts before ``10``."""
    name = Path(value).name
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


@dataclass
class SequenceSpec:
    """One offline sequence: paired videos or paired ordered frame directories.

    Attributes:
        name: Sequence name used in result metadata and output file names.
        kind: ``"video"`` or ``"frames"``.
        fps: Sequence frame rate.
        length: Number of frames. Zero means the video frame count was unavailable.
        slots: Contract slot letters, e.g. ``("A", "B")``.
        source_paths: Slot -> video path (video) or frame-directory root (frames); None for missing slots.
        frame_paths: Slot -> ordered frame paths (frames only).
    """

    name: str
    kind: str
    fps: float
    length: int
    slots: Tuple[str, ...]
    source_paths: Dict[str, Optional[Path]] = field(default_factory=dict)
    frame_paths: Dict[str, List[Path]] = field(default_factory=dict)

    @property
    def anchor_slot(self) -> str:
        """Return the first slot that has a real source."""
        for slot in self.slots:
            if self.source_paths.get(slot) is not None:
                return slot
        raise ValueError(f"序列 {self.name!r} 没有任何有效槽位")

    def to_sample_specs(self) -> List[Dict]:
        """Expand a frame-directory sequence into ordered dataset sample specs."""
        if self.kind != SEQUENCE_KIND_FRAMES:
            raise ValueError(f"序列 {self.name!r} 是 {self.kind} 序列，不能展开为逐帧样本")
        samples = []
        for idx in range(self.length):
            spec = {"id": f"{self.name}_{idx + 1:06d}", "seq": self.name, "frame": idx + 1, "fps": self.fps}
            for slot in self.slots:
                frames = self.frame_paths.get(slot) or []
                spec[slot] = frames[idx] if idx < len(frames) else None
            samples.append(spec)
        return samples


@dataclass
class _SlotSource:
    """Internal classification result for one source slot."""

    form: str
    video: Optional[Path] = None
    frames: List[Path] = field(default_factory=list)
    sequences: Dict[str, Path] = field(default_factory=dict)


def _visible_children(path: Path) -> List[Path]:
    """List non-hidden direct children in natural order."""
    return sorted((p for p in path.iterdir() if not p.name.startswith(".")), key=natural_sort_key)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_SUFFIXES


def _normalize_frame_files(slot: str, files: List[Path]) -> List[Path]:
    """Validate frame files, reject duplicate stems, and return them in natural order."""
    stems = {}
    for path in files:
        if not _is_image(path):
            raise ValueError(f"槽位 {slot!r} 目录 {path.parent} 含非图片文件: {path.name}")
        if path.stem in stems:
            raise ValueError(
                f"槽位 {slot!r} 目录 {path.parent} 存在重名 stem: {path.stem!r} "
                f"({stems[path.stem].name!r} 与 {path.name!r})"
            )
        stems[path.stem] = path
    return sorted(files, key=natural_sort_key)


def _scan_frame_directory(slot: str, directory: Path) -> List[Path]:
    """Validate a frame-directory leaf and return its ordered image paths."""
    entries = _visible_children(directory)
    nested = [p for p in entries if p.is_dir()]
    if nested:
        raise ValueError(f"槽位 {slot!r} 帧目录 {directory} 含嵌套子目录: {nested[0].name!r}")
    files = [p for p in entries if p.is_file()]
    if not files:
        raise ValueError(f"槽位 {slot!r} 帧目录 {directory} 内没有图片")
    return _normalize_frame_files(slot, files)


class SequenceResolver:
    """Resolve tracking sources into ordered offline sequences.

    Supported source forms (all provided slots must use the same form):
    - one video file -> one video sequence
    - a directory of images -> one frame sequence
    - a directory of videos -> multiple video sequences
    - a directory of image subdirectories -> multiple frame sequences
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def resolve(
        self,
        sources: Dict[str, Union[str, Path, None]],
        contract,
        strict_match: bool = True,
        fps: Optional[float] = None,
    ) -> List[SequenceSpec]:
        """Resolve slot sources into sequences.

        Args:
            sources: Slot letter -> video file, image directory, video directory, or image-subdirectory root.
            contract: ``MultiModalChannelContract`` providing slot identity.
            strict_match: Require identical sequence/stem sets across slots; otherwise take the intersection.
            fps: Fallback frame rate for frame directories (default 30) and unreadable video metadata.

        Returns:
            Ordered list of ``SequenceSpec``.
        """
        if fps is not None and (not np.isfinite(fps) or fps <= 0):
            raise ValueError(f"fps 必须是正数，收到 {fps!r}")
        slots, normalized = self._normalize_sources(sources, contract)
        if all(value is None for value in normalized.values()):
            raise ValueError("sources 全部为 None，至少提供一个槽位的视频/帧目录")

        inspected: Dict[str, _SlotSource] = {}
        for slot in slots:
            path = normalized.get(slot)
            if path is not None:
                inspected[slot] = self._inspect_source(slot, path)

        forms = {slot: source.form for slot, source in inspected.items()}
        if len(set(forms.values())) > 1:
            detail = ", ".join(f"{slot}={form}" for slot, form in forms.items())
            raise ValueError(f"各槽位输入形态必须一致（视频文件/帧目录/视频目录/帧序列根目录）：{detail}")
        form = next(iter(forms.values()))

        frame_fps = float(fps) if fps is not None else 30.0
        if form == "video":
            sequences = [self._build_single_video(slots, inspected, normalized, fps)]
        elif form == "frames":
            sequences = [self._build_single_frames(slots, inspected, normalized, frame_fps, strict_match)]
        elif form == "multi_video":
            sequences = self._build_multi_videos(slots, inspected, normalized, fps, strict_match)
        else:
            sequences = self._build_multi_frames(slots, inspected, normalized, frame_fps, strict_match)

        self._check_unique_names(sequences)
        if self.verbose:
            LOGGER.info(f"序列解析完成: {len(sequences)} 个序列, 形态 {form}")
            for seq in sequences:
                LOGGER.info(f"  {seq.name}: {seq.kind}, {seq.length} 帧, {seq.fps:g} fps")
        return sequences

    def _normalize_sources(self, sources, contract) -> Tuple[Tuple[str, ...], Dict[str, Optional[Path]]]:
        if not isinstance(sources, dict) or not sources:
            raise ValueError("sources 必须是非空字典（槽位字母 -> 视频/目录）")
        normalized: Dict[str, Optional[Path]] = {}
        for key, value in sources.items():
            slot = str(key).strip().upper()
            if len(slot) != 1 or not slot.isalpha():
                raise ValueError(f"sources 键 {key!r} 必须是单个槽位字母（A-Z）")
            if contract is not None and slot not in contract.slots:
                raise ValueError(f"sources 键 {slot!r} 不在契约槽位 {list(contract.slots)} 中")
            if value is None:
                normalized[slot] = None
                continue
            if isinstance(value, (list, tuple)):
                raise ValueError(
                    f"sources[{slot!r}] 不支持图片/视频路径列表；请使用视频文件、帧目录或多序列根目录"
                )
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"sources[{slot!r}] 路径不存在: {path}")
            normalized[slot] = path

        slots = tuple(contract.slots) if contract is not None else tuple(sorted(normalized))
        for slot in slots:
            normalized.setdefault(slot, None)
        extra = set(normalized) - set(slots)
        if extra:
            raise ValueError(f"sources 含契约外槽位: {sorted(extra)}")
        return slots, normalized

    def _inspect_source(self, slot: str, path: Path) -> _SlotSource:
        if path.is_file():
            if not _is_video(path):
                raise ValueError(
                    f"槽位 {slot!r} 文件 {path} 不是视频；单张图片不能作为序列输入，请使用视频或帧目录"
                )
            return _SlotSource("video", video=path)

        if not path.is_dir():
            raise ValueError(f"槽位 {slot!r} 路径既不是视频文件也不是目录: {path}")

        entries = _visible_children(path)
        files = [p for p in entries if p.is_file()]
        dirs = [p for p in entries if p.is_dir()]
        if files and dirs:
            raise ValueError(f"槽位 {slot!r} 目录 {path} 同时包含文件和子目录，无法判定序列形态")
        if files:
            image_files = [p for p in files if _is_image(p)]
            video_files = [p for p in files if _is_video(p)]
            if len(image_files) == len(files):
                return _SlotSource("frames", frames=_normalize_frame_files(slot, files))
            if len(video_files) == len(files):
                return _SlotSource("multi_video", sequences=self._index_files(slot, video_files, "视频"))
            bad = [p.name for p in files if p not in image_files and p not in video_files]
            if bad:
                raise ValueError(f"槽位 {slot!r} 目录 {path} 含不支持的文件: {bad[:5]}")
            raise ValueError(f"槽位 {slot!r} 目录 {path} 同时包含图片和视频文件，无法判定序列形态")
        if dirs:
            for directory in dirs:
                _scan_frame_directory(slot, directory)
            return _SlotSource("multi_frames", sequences={d.name: d for d in dirs})
        raise ValueError(f"槽位 {slot!r} 目录 {path} 为空")

    @staticmethod
    def _index_files(slot: str, files: List[Path], label: str) -> Dict[str, Path]:
        indexed = {}
        for path in files:
            if path.stem in indexed:
                raise ValueError(f"槽位 {slot!r} 存在重名{label} stem: {path.stem!r}")
            indexed[path.stem] = path
        return indexed

    def _build_single_video(self, slots, inspected, normalized, fps) -> SequenceSpec:
        anchor = next(iter(inspected))
        paths = {slot: source.video for slot, source in inspected.items()}
        seq_fps, length = self._video_metadata(paths, fps)
        return SequenceSpec(
            name=paths[anchor].stem,
            kind=SEQUENCE_KIND_VIDEO,
            fps=seq_fps,
            length=length,
            slots=slots,
            source_paths={slot: normalized.get(slot) for slot in slots},
        )

    def _build_single_frames(self, slots, inspected, normalized, fps, strict_match) -> SequenceSpec:
        anchor = next(iter(inspected))
        frame_lists = {slot: source.frames for slot, source in inspected.items()}
        name = self._frame_sequence_name(
            normalized[anchor], [normalized[s] for s in inspected if s != anchor]
        )
        frame_paths, length = self._pair_frames(frame_lists, anchor, strict_match, name)
        return SequenceSpec(
            name=name,
            kind=SEQUENCE_KIND_FRAMES,
            fps=fps,
            length=length,
            slots=slots,
            source_paths={slot: normalized.get(slot) for slot in slots},
            frame_paths=frame_paths,
        )

    def _build_multi_videos(self, slots, inspected, normalized, fps, strict_match) -> List[SequenceSpec]:
        anchor = next(iter(inspected))
        names = self._pair_sequence_names(inspected, anchor, strict_match, "视频")
        sequences = []
        for name in names:
            paths = {slot: source.sequences[name] for slot, source in inspected.items()}
            seq_fps, length = self._video_metadata(paths, fps)
            sequences.append(
                SequenceSpec(
                    name=name,
                    kind=SEQUENCE_KIND_VIDEO,
                    fps=seq_fps,
                    length=length,
                    slots=slots,
                    source_paths={slot: paths.get(slot) for slot in slots},
                )
            )
        return sequences

    def _build_multi_frames(self, slots, inspected, normalized, fps, strict_match) -> List[SequenceSpec]:
        anchor = next(iter(inspected))
        names = self._pair_sequence_names(inspected, anchor, strict_match, "帧序列")
        sequences = []
        for name in names:
            directories = {slot: source.sequences[name] for slot, source in inspected.items()}
            frame_lists = {
                slot: _scan_frame_directory(slot, directory) for slot, directory in directories.items()
            }
            seq_name = self._frame_sequence_name(
                directories[anchor], [directories[s] for s in inspected if s != anchor]
            )
            frame_paths, length = self._pair_frames(frame_lists, anchor, strict_match, seq_name)
            sequences.append(
                SequenceSpec(
                    name=seq_name,
                    kind=SEQUENCE_KIND_FRAMES,
                    fps=fps,
                    length=length,
                    slots=slots,
                    source_paths={slot: directories.get(slot) for slot in slots},
                    frame_paths=frame_paths,
                )
            )
        return sequences

    def _pair_sequence_names(self, inspected: Dict[str, _SlotSource], anchor: str, strict_match: bool, label: str):
        base = list(inspected[anchor].sequences)
        if strict_match:
            base_set = set(base)
            for slot, source in inspected.items():
                names = set(source.sequences)
                if names != base_set:
                    missing = sorted(base_set - names, key=natural_sort_key)[:5]
                    extra = sorted(names - base_set, key=natural_sort_key)[:5]
                    raise ValueError(
                        f"槽位 {slot!r} 与锚定槽位 {anchor!r} 的{label}名集合不一致；"
                        f"缺失示例 {missing}，多余示例 {extra}"
                    )
            return sorted(base, key=natural_sort_key)

        common = set.intersection(*(set(source.sequences) for source in inspected.values()))
        skipped = len(base) - len(common)
        if skipped:
            LOGGER.warning(f"strict_match=False：跳过 {skipped} 个无法在全部槽位配对的{label}")
        if not common:
            raise ValueError(f"strict_match=False：各槽位之间没有共同的{label}名")
        return [name for name in base if name in common]

    def _pair_frames(
        self,
        frame_lists: Dict[str, List[Path]],
        anchor: str,
        strict_match: bool,
        seq_name: str,
    ) -> Tuple[Dict[str, List[Path]], int]:
        stem_maps = {
            slot: {path.stem: path for path in frames} for slot, frames in frame_lists.items()
        }
        if strict_match:
            base_stems = set(stem_maps[anchor])
            for slot, stems in stem_maps.items():
                if set(stems) != base_stems:
                    missing = sorted(base_stems - set(stems), key=natural_sort_key)[:5]
                    extra = sorted(set(stems) - base_stems, key=natural_sort_key)[:5]
                    raise ValueError(
                        f"序列 {seq_name!r} 槽位 {slot!r} 与锚定槽位 {anchor!r} 的帧 stem 集合不一致；"
                        f"缺失示例 {missing}，多余示例 {extra}"
                    )
            ordered_stems = [path.stem for path in frame_lists[anchor]]
        else:
            common = set.intersection(*(set(stems) for stems in stem_maps.values()))
            ordered_stems = [path.stem for path in frame_lists[anchor] if path.stem in common]
            skipped = len(frame_lists[anchor]) - len(ordered_stems)
            if skipped:
                LOGGER.warning(f"strict_match=False：序列 {seq_name!r} 跳过 {skipped} 帧无法配对的帧")
            if not ordered_stems:
                raise ValueError(f"strict_match=False：序列 {seq_name!r} 各槽位之间没有共同帧")

        frame_paths = {slot: [stems[stem] for stem in ordered_stems] for slot, stems in stem_maps.items()}
        return frame_paths, len(ordered_stems)

    @staticmethod
    def _frame_sequence_name(anchor_dir: Path, other_dirs: Sequence[Path]) -> str:
        dirs = [anchor_dir, *other_dirs]
        if all(directory.name == anchor_dir.name for directory in dirs):
            return anchor_dir.name
        if len({directory.parent.resolve() for directory in dirs}) == 1:
            return anchor_dir.parent.name
        return anchor_dir.name

    @staticmethod
    def _video_metadata(paths: Dict[str, Path], fps_override: Optional[float]) -> Tuple[float, int]:
        fps_values = {}
        lengths = {}
        for slot, path in paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                raise ValueError(f"槽位 {slot!r} 无法打开视频: {path}")
            try:
                raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            if not np.isfinite(raw_fps) or raw_fps <= 0:
                if fps_override is None:
                    raise ValueError(f"槽位 {slot!r} 视频帧率不可读，请显式传入 fps: {path}")
                raw_fps = float(fps_override)
            fps_values[slot] = raw_fps
            lengths[slot] = count

        fps_list = list(fps_values.values())
        # Variable-frame-rate containers can report slightly different average fps across slots.
        if max(fps_list) - min(fps_list) > 0.01:
            raise ValueError(f"各槽位视频帧率不一致: {fps_values}")
        known = {slot: count for slot, count in lengths.items() if count > 0}
        if len(known) != len(lengths):
            LOGGER.warning(f"部分视频帧数不可读，跳过帧数预检: {lengths}")
        if known and len(set(known.values())) > 1:
            raise ValueError(f"各槽位视频帧数不一致: {lengths}")
        length = next(iter(known.values())) if known else 0
        return fps_list[0], length

    @staticmethod
    def _check_unique_names(sequences: List[SequenceSpec]) -> None:
        seen = set()
        for seq in sequences:
            if seq.name in seen:
                raise ValueError(f"序列名重复: {seq.name!r}；请使用不同的文件/目录名区分序列")
            seen.add(seq.name)


class MultiModalVideoDataset(MultiModalInferenceDataset):
    """Sequential decoder for one paired-video ``SequenceSpec``.

    Each iteration opens all slot videos, decodes them in lockstep, and reuses
    ``MultiModalInferenceDataset._build_sample`` for alignment and tensorization.
    """

    def __init__(self, sequence: SequenceSpec, imgsz, dataset_config, stride: int = 32, pad: float = 0.5,
                 verbose: bool = True):
        if sequence.kind != SEQUENCE_KIND_VIDEO:
            raise ValueError(f"MultiModalVideoDataset 只接受 video 序列，收到 {sequence.kind!r}")
        super().__init__(
            samples=[],
            imgsz=imgsz,
            dataset_config=dataset_config,
            stride=stride,
            pad=pad,
            verbose=False,
        )
        self.sequence = sequence
        self.ni = sequence.length
        self.anchor_slot = sequence.anchor_slot
        if verbose:
            LOGGER.info(
                f"MultiModalVideoDataset 初始化完成: 序列 {sequence.name}, "
                f"帧数 {sequence.length}, {sequence.fps:g} fps"
            )

    def __len__(self) -> int:
        return self.sequence.length

    def __getitem__(self, index: int) -> Dict:
        raise TypeError("MultiModalVideoDataset 只支持顺序迭代，不支持按索引访问")

    def __iter__(self):
        seq = self.sequence
        slot_paths = {slot: path for slot in seq.slots if (path := seq.source_paths.get(slot)) is not None}
        caps = {}
        try:
            for slot, path in slot_paths.items():
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    cap.release()
                    raise ValueError(f"无法打开槽位 {slot!r} 视频: {path}")
                caps[slot] = cap

            frame = 0
            while True:
                slot_imgs = {}
                failed = []
                for slot, cap in caps.items():
                    ok, img = cap.read()
                    if not ok:
                        failed.append(slot)
                    else:
                        slot_imgs[slot] = img
                if failed:
                    if len(failed) == len(caps):
                        break
                    raise ValueError(f"序列 {seq.name!r} 槽位 {failed} 在第 {frame + 1} 帧提前结束")
                frame += 1
                spec = {slot: seq.source_paths.get(slot) for slot in seq.slots}
                spec.update({"id": f"{seq.name}_{frame:06d}", "seq": seq.name, "frame": frame, "fps": seq.fps})
                yield self._build_sample(spec, slot_imgs, self.anchor_slot)

            if seq.length and frame != seq.length:
                raise ValueError(f"序列 {seq.name!r} 解码帧数 {frame} 与预检帧数 {seq.length} 不一致")
        finally:
            for cap in caps.values():
                cap.release()
