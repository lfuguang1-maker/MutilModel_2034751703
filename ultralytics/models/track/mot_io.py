# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MOT-format trajectory I/O shared by TRACK predictions and YOLOMMtrack ground truth."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

_COORD_PRECISION = 6


@dataclass(frozen=True)
class MotRecord:
    """One trajectory row: frame, track id, coordinates, score/flag, class, visibility."""

    frame: int
    track_id: int
    coords: Tuple[float, ...]  # (x, y, w, h) for box tasks, 8 corner values for obb
    score: float
    cls: int
    vis: float = -1.0


def _fmt(value: float) -> str:
    return f"{float(value):.{_COORD_PRECISION}f}"


def format_result_lines(result, task: str = "detect") -> List[str]:
    """Format one tracked result frame as MOT lines (only entries with an id)."""
    if result.ids is None:
        return []
    ids = list(result.ids)
    frame = int(result.meta.get("frame", 0))
    if frame <= 0:
        raise ValueError(f"轨迹帧号必须从 1 开始，收到 {frame}")
    if not ids:
        return []

    lines = []
    if task == "obb":
        corners = result.obb.xyxyxyxy
        if len(ids) != len(corners):
            raise ValueError(f"OBB 结果与编号数量不一致: {len(corners)} boxes vs {len(ids)} ids")
        for i, track_id in enumerate(ids):
            points = [_fmt(v) for v in corners[i].reshape(-1)]
            lines.append(
                f"{frame},{int(track_id)}," + ",".join(points)
                + f",{_fmt(result.obb.conf[i])},{int(result.obb.cls[i])}"
            )
        return lines

    boxes = result.boxes
    if len(ids) != len(boxes):
        raise ValueError(f"结果与编号数量不一致: {len(boxes)} boxes vs {len(ids)} ids")
    for i, track_id in enumerate(ids):
        x1, y1, x2, y2 = (float(v) for v in boxes[i, :4])
        lines.append(
            f"{frame},{int(track_id)},{_fmt(x1)},{_fmt(y1)},{_fmt(x2 - x1)},{_fmt(y2 - y1)},"
            f"{_fmt(boxes[i, 4])},{int(boxes[i, 5])},-1"
        )
    return lines


def write_results(path: Union[str, Path], results: Iterable, task: str = "detect") -> None:
    """Write tracked results to a MOT trajectory file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            for line in format_result_lines(result, task=task):
                f.write(line + "\n")


def read_mot(path: Union[str, Path], task: str = "detect", nc: Optional[int] = None) -> List[MotRecord]:
    """Read and validate a MOT trajectory file.

    Box tasks accept 8 or 9 columns; obb requires 12 columns. Frames and ids must be
    positive integers and classes must be non-negative (and below ``nc`` when given).
    """
    path = Path(path)
    records: List[MotRecord] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            if task == "obb":
                if len(parts) != 12:
                    raise ValueError(f"OBB 轨迹应为 12 列，实际 {len(parts)} 列")
                frame, track_id = int(parts[0]), int(parts[1])
                coords = tuple(float(v) for v in parts[2:10])
                score, cls = float(parts[10]), int(parts[11])
                vis = -1.0
            else:
                if len(parts) not in (8, 9):
                    raise ValueError(f"框轨迹应为 8 或 9 列，实际 {len(parts)} 列")
                frame, track_id = int(parts[0]), int(parts[1])
                coords = tuple(float(v) for v in parts[2:6])
                score, cls = float(parts[6]), int(parts[7])
                vis = float(parts[8]) if len(parts) == 9 else -1.0
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: 无法解析轨迹行 ({raw!r}): {e}") from None

        if frame <= 0:
            raise ValueError(f"{path}:{lineno}: frame 必须为正整数，收到 {frame}")
        if track_id <= 0:
            raise ValueError(f"{path}:{lineno}: id 必须为正整数，收到 {track_id}")
        if cls < 0 or (nc is not None and cls >= nc):
            raise ValueError(f"{path}:{lineno}: cls 越界，收到 {cls}（nc={nc}）")
        records.append(MotRecord(frame=frame, track_id=track_id, coords=coords, score=score, cls=cls, vis=vis))
    return records


def write_mot(path: Union[str, Path], records: Sequence[MotRecord], task: str = "detect") -> None:
    """Write MOT records (predictions or ground truth) to a trajectory file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            if task == "obb":
                if len(rec.coords) != 8:
                    raise ValueError(f"OBB 记录需要 8 个坐标值，收到 {len(rec.coords)}")
                values = [
                    str(rec.frame), str(rec.track_id), *(_fmt(v) for v in rec.coords),
                    _fmt(rec.score), str(rec.cls),
                ]
            else:
                if len(rec.coords) != 4:
                    raise ValueError(f"框记录需要 4 个坐标值，收到 {len(rec.coords)}")
                values = [
                    str(rec.frame),
                    str(rec.track_id),
                    *(_fmt(v) for v in rec.coords),
                    _fmt(rec.score),
                    str(rec.cls),
                    _fmt(rec.vis),
                ]
            f.write(",".join(values) + "\n")
