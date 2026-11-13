# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-sequence annotated video and MOT trajectory output for TRACK."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .mot_io import format_result_lines


def _trail_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR color for a track id."""
    return (37 * track_id % 255, (97 * track_id + 80) % 255, (17 * track_id + 160) % 255)


def _centers(result) -> Dict[int, Tuple[float, float]]:
    """Return track-id -> center point for the current frame."""
    if result.ids is None or len(result.ids) == 0:
        return {}
    if hasattr(result, "obb"):
        xywhr = result.obb.xywhr
        return {int(tid): (float(xywhr[i, 0]), float(xywhr[i, 1])) for i, tid in enumerate(result.ids)}
    boxes = result.boxes
    return {
        int(tid): ((float(boxes[i, 0]) + float(boxes[i, 2])) / 2.0, (float(boxes[i, 1]) + float(boxes[i, 3])) / 2.0)
        for i, tid in enumerate(result.ids)
    }


class TrackSaver:
    """Write one annotated video and one trajectory file per sequence."""

    def __init__(
        self,
        save_dir: Path,
        task: str,
        save: bool = True,
        save_txt: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        show_conf: bool = True,
        show_labels: bool = True,
        trail: int = 30,
    ):
        self.save_dir = Path(save_dir)
        self.task = task
        self.save = save
        self.save_txt = save_txt
        self.line_width = line_width
        self.font_size = font_size
        self.show_conf = show_conf
        self.show_labels = show_labels
        self.trail = max(0, int(trail))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.save_txt:
            (self.save_dir / "labels").mkdir(parents=True, exist_ok=True)
        self._seq = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._txt = None
        self._trails: Dict[int, list] = {}
        self._written = 0

    def start_sequence(self, seq) -> None:
        """Reset per-sequence state and trails."""
        self._seq = seq
        self._writer = None
        self._txt = None
        self._trails = {}
        self._written = 0

    def write(self, result, seq) -> None:
        """Write one tracked frame to the trajectory file and annotated video."""
        if self.save_txt:
            if self._txt is None:
                self._txt = (self.save_dir / "labels" / f"{seq.name}.txt").open("w", encoding="utf-8")
            for line in format_result_lines(result, task=self.task):
                self._txt.write(line + "\n")

        if self.save:
            frame_img = self._annotated_frame(result)
            if self._writer is None:
                height, width = frame_img.shape[:2]
                path = self.save_dir / f"{seq.name}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(seq.fps), (width, height)
                )
                if not writer.isOpened():
                    writer.release()
                    raise ValueError(
                        f"无法打开视频写入器: {path}。请确认 OpenCV 编码器可用，或改用 .avi 输出。"
                    )
                self._writer = writer
            self._writer.write(frame_img)
        self._written += 1

    def close_sequence(self, seq, verify: bool = True) -> None:
        """Release per-sequence resources and verify the written frame count."""
        if self._txt is not None:
            self._txt.close()
            self._txt = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if verify and self.save and seq.length and self._written != seq.length:
            raise ValueError(f"序列 {seq.name!r} 写入帧数 {self._written} 与预期 {seq.length} 不一致")
        self._seq = None

    def close(self) -> None:
        """Release any remaining resources."""
        if self._txt is not None:
            self._txt.close()
            self._txt = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _annotated_frame(self, result) -> np.ndarray:
        """Plot all real slot panels, draw trails, and concatenate horizontally."""
        plotted = result.plot(
            conf=self.show_conf,
            line_width=self.line_width,
            font_size=self.font_size,
            labels=self.show_labels,
            show_filename=False,
        )
        slots = [slot for slot in result.slots if result.orig_imgs.get(slot) is not None]
        panels = [plotted[slot] for slot in slots if slot in plotted]
        if not panels:
            raise ValueError(f"结果没有可绘制的槽位: {slots}")

        if self.trail:
            self._update_trails(result)
        for panel in panels:
            self._draw_trails(panel)

        height = panels[0].shape[0]
        if any(panel.shape[0] != height for panel in panels):
            raise ValueError("各槽位标注图高度不一致，无法横向拼接")
        frame_img = panels[0] if len(panels) == 1 else np.hstack(panels)
        return np.ascontiguousarray(frame_img)

    def _update_trails(self, result) -> None:
        for track_id, center in _centers(result).items():
            points = self._trails.setdefault(track_id, [])
            points.append(center)
            if len(points) > self.trail:
                del points[0 : len(points) - self.trail]

    def _draw_trails(self, img: np.ndarray) -> None:
        if not self.trail:
            return
        for track_id, points in self._trails.items():
            if len(points) < 2:
                continue
            color = _trail_color(track_id)
            for p1, p2 in zip(points, points[1:]):
                cv2.line(
                    img,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    color,
                    2,
                    cv2.LINE_AA,
                )
