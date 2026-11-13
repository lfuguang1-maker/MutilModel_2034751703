# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOMMtrack dataset evaluation for TRACK."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from ultralytics.cfg import get_save_dir
from ultralytics.data.multimodal import SequenceResolver
from ultralytics.models.track.mot_io import MotRecord, read_mot
from ultralytics.models.track.runner import run_tracking
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.mot_metrics import (
    CLEAR,
    HOTA,
    Identity,
    apply_ignore_regions,
    box_iou_matrix,
    polygon_iou_matrix,
    remap_ids,
)
from ultralytics.utils.val_results import TrackValResults


def _normalize_names(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"names 必须是 dict 或 list，收到 {type(names).__name__}")


def _resolve_dataset(data, split: str, contract, model_names: Dict[int, str], modality: Optional[str]):
    yaml_path = Path(check_yaml(data))
    cfg = YAML.load(yaml_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"数据集 YAML 内容无效: {yaml_path}")
    root = Path(cfg.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    names = _normalize_names(cfg.get("names"))
    if names != {int(k): str(v) for k, v in model_names.items()}:
        raise ValueError(f"数据集 names 与权重 names 不一致: {names} != {model_names}")
    if "nc" in cfg and int(cfg["nc"]) != len(names):
        raise ValueError(f"数据集 nc={cfg['nc']} 与 names 数量 {len(names)} 不一致")

    split_value = cfg.get(split)
    if not split_value:
        raise ValueError(f"数据集 YAML 缺少 split {split!r}: {yaml_path}")
    modality_cfg = cfg.get("modality") or {}
    contract_names = [str(n) for n in contract.modality_names]
    if any(name not in modality_cfg for name in contract_names):
        # YAML-built checkpoints can carry generic names (modality_A/B); use data.yaml's modality_used.
        used = cfg.get("modality_used")
        if (isinstance(used, (list, tuple)) and len(used) == len(contract_names)
                and all(str(name) in modality_cfg for name in used)):
            contract_names = [str(name) for name in used]
        else:
            missing = [name for name in contract_names if name not in modality_cfg]
            raise ValueError(f"数据集 modality 缺少权重契约模态 {missing}: {list(modality_cfg)}")
    if modality is not None and modality not in contract_names:
        raise ValueError(f"modality={modality!r} 不在权重契约 {contract_names} 中")
    if not cfg.get("tracks"):
        raise ValueError("TRACK.val 需要 data.yaml 中的 tracks 键")

    split_path = Path(str(split_value))
    anchor_dir = str(modality_cfg[contract_names[0]])
    if split_path.parts and split_path.parts[0] == anchor_dir:
        split_rel = Path(*split_path.parts[1:])
    else:
        split_rel = split_path

    sources = {}
    for slot, name in zip(contract.slots, contract_names):
        if modality is not None and name != modality:
            sources[slot] = None
            continue
        source = root / str(modality_cfg[name]) / split_rel
        if not source.exists():
            raise FileNotFoundError(f"模态 {name!r} 的 {split} 路径不存在: {source}")
        sources[slot] = source

    tracks_dir = root / str(cfg["tracks"]) / split_rel
    if not tracks_dir.is_dir():
        raise FileNotFoundError(f"轨迹真值目录不存在: {tracks_dir}")
    fps = cfg.get("fps", 30)
    return cfg, sources, tracks_dir, fps


def _seq_frames(seq, records: List[MotRecord]) -> Dict[int, List[MotRecord]]:
    grouped: Dict[int, List[MotRecord]] = {}
    for record in records:
        if seq.length and record.frame > seq.length:
            raise ValueError(f"序列 {seq.name!r} 真值帧号 {record.frame} 超过实际帧数 {seq.length}")
        grouped.setdefault(record.frame, []).append(record)
    return grouped


def _box_xyxy(record: MotRecord) -> np.ndarray:
    x, y, w, h = record.coords
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def _obb_poly(record: MotRecord) -> np.ndarray:
    return np.asarray(record.coords, dtype=np.float32).reshape(4, 2)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _merge_metrics(hota: Dict, clear: Dict, identity: Dict) -> Dict:
    merged = {}
    merged.update(hota)
    merged.update(clear)
    merged.update(identity)
    return merged


def _print_table(per_class: Dict[str, Dict], class_avg: Dict, det_avg: Dict) -> None:
    columns = ["HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDP", "IDR", "CLR_FP", "CLR_FN", "IDSW"]
    LOGGER.info("Class".ljust(18) + "".join(f"{c:>9}" for c in columns))
    for name, res in per_class.items():
        row = f"{name[:17]:<18}"
        for col in columns:
            value = res.get(col, 0.0)
            value = float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
            row += f"{value:>9.3f}"
        LOGGER.info(row)
    for label, res in (("CLASS_AVG", class_avg), ("DET_AVG", det_avg)):
        row = f"{label:<18}"
        for col in columns:
            value = res.get(col, 0.0)
            value = float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
            row += f"{value:>9.3f}"
        LOGGER.info(row)


def run_track_val(model, data, **params) -> TrackValResults:
    """Evaluate TRACK predictions against a YOLOMMtrack dataset."""
    split = params.get("split", "val")
    tracker_cfg = model._load_tracker_cfg(params.get("tracker", "bytetrack.yaml"))
    engine = model.build_engine(
        conf=params.get("conf", 0.1),
        iou=params.get("iou"),
        imgsz=params.get("imgsz"),
        max_det=params.get("max_det"),
        device=params.get("device"),
    )
    contract = getattr(engine.router, "contract", None)
    if contract is None:
        raise RuntimeError("TRACK.val 需要带通道契约的多模态引擎")
    cfg, sources, tracks_dir, fps = _resolve_dataset(
        data, split, contract, model.names, params.get("modality")
    )
    sequences = SequenceResolver(verbose=params.get("verbose", True)).resolve(
        sources, contract, strict_match=True, fps=fps
    )

    expected = {f"{seq.name}.txt" for seq in sequences}
    actual = {p.name for p in tracks_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
    missing, extra = expected - actual, actual - expected
    if missing:
        raise FileNotFoundError(f"轨迹真值缺失: {sorted(missing)} in {tracks_dir}")
    if extra:
        raise ValueError(f"轨迹真值目录含多余文件: {sorted(extra)} in {tracks_dir}")

    truth = {seq.name: read_mot(tracks_dir / f"{seq.name}.txt", task=model.task, nc=len(model.names))
             for seq in sequences}
    for records in truth.values():
        flags = {float(r.score) for r in records}
        if not flags.issubset({0.0, 1.0}):
            raise ValueError(f"真值 score/flag 列只能是 0 或 1，收到 {sorted(flags)}")

    save_dir = get_save_dir(
        SimpleNamespace(
            task=model.task,
            mode="track_val",
            project=params.get("project"),
            name=params.get("name"),
            exist_ok=params.get("exist_ok", False),
        )
    )
    predictions: Dict[str, List[MotRecord]] = {seq.name: [] for seq in sequences}
    results = run_tracking(
        model,
        sources=sources,
        tracker_cfg=tracker_cfg,
        stream=True,
        fps=fps,
        strict_match=True,
        classes=params.get("classes"),
        conf=params.get("conf", 0.1),
        iou=params.get("iou"),
        imgsz=params.get("imgsz"),
        max_det=params.get("max_det"),
        device=params.get("device"),
        save=params.get("save", False),
        save_txt=params.get("save_txt", False),
        save_dir=save_dir if (params.get("save") or params.get("save_txt")) else None,
        verbose=params.get("verbose", True),
    )
    for result in results:
        seq_name = result.meta["seq"]
        frame = int(result.meta["frame"])
        ids = [] if result.ids is None else list(result.ids)
        if model.task == "obb":
            corners = result.obb.xyxyxyxy
            for i, track_id in enumerate(ids):
                predictions[seq_name].append(
                    MotRecord(frame, int(track_id), tuple(float(v) for v in corners[i].reshape(-1)),
                              float(result.obb.conf[i]), int(result.obb.cls[i]))
                )
        else:
            for i, track_id in enumerate(ids):
                x1, y1, x2, y2 = (float(v) for v in result.boxes[i, :4])
                predictions[seq_name].append(
                    MotRecord(frame, int(track_id), (x1, y1, x2 - x1, y2 - y1),
                              float(result.boxes[i, 4]), int(result.boxes[i, 5]))
                )

    frame_items: Dict[str, List] = {}
    seen_classes = set()
    for seq in sequences:
        gt_by_frame = _seq_frames(seq, truth[seq.name])
        tr_by_frame = _seq_frames(seq, predictions[seq.name])
        n_frames = seq.length or max([0, *gt_by_frame.keys(), *tr_by_frame.keys()])
        items = []
        for frame in range(1, n_frames + 1):
            gt = gt_by_frame.get(frame, [])
            tr = tr_by_frame.get(frame, [])
            if model.task == "obb":
                gt_geom = np.array([_obb_poly(r) for r in gt], dtype=np.float32).reshape(-1, 4, 2)
                tr_geom = np.array([_obb_poly(r) for r in tr], dtype=np.float32).reshape(-1, 4, 2)
                similarity_fn = polygon_iou_matrix
            else:
                gt_geom = np.array([_box_xyxy(r) for r in gt], dtype=np.float64).reshape(-1, 4)
                tr_geom = np.array([_box_xyxy(r) for r in tr], dtype=np.float64).reshape(-1, 4)
                similarity_fn = box_iou_matrix
            ignore_mask = np.array([int(round(r.score)) == 0 for r in gt], dtype=bool)
            keep_gt, keep_tr = apply_ignore_regions(gt_geom, tr_geom, ignore_mask, similarity_fn)
            gt = [r for r, keep in zip(gt, keep_gt) if keep]
            tr = [r for i, r in enumerate(tr) if i in set(keep_tr.tolist())]
            gt_geom, tr_geom = gt_geom[keep_gt], tr_geom[keep_tr]
            seen_classes.update(r.cls for r in gt)
            seen_classes.update(r.cls for r in tr)
            items.append((gt, tr, gt_geom, tr_geom))
        frame_items[seq.name] = items

    requested = params.get("classes")
    if requested is not None:
        requested = {int(c) for c in np.atleast_1d(requested).tolist()}
        classes = sorted(seen_classes & requested)
    else:
        classes = sorted(seen_classes)
    if not classes:
        raise ValueError("没有可评估的类别：truth 与 predictions 均为空或 classes 过滤后为空")

    per_class: Dict[str, Dict] = {}
    per_sequence: Dict[str, Dict] = {}
    for cls_id in classes:
        class_name = str(model.names.get(int(cls_id), int(cls_id)))
        seq_results = {}
        for seq in sequences:
            gt_ids, tr_ids, sims = [], [], []
            for gt, tr, gt_geom, tr_geom in frame_items[seq.name]:
                gt_c = [r for r in gt if r.cls == cls_id]
                tr_c = [r for r in tr if r.cls == cls_id]
                gt_ids.append(np.array([r.track_id for r in gt_c], dtype=int))
                tr_ids.append(np.array([r.track_id for r in tr_c], dtype=int))
                if model.task == "obb":
                    g_idx = [i for i, r in enumerate(gt) if r.cls == cls_id]
                    t_idx = [i for i, r in enumerate(tr) if r.cls == cls_id]
                    sims.append(polygon_iou_matrix(gt_geom[g_idx], tr_geom[t_idx]))
                else:
                    g_idx = [i for i, r in enumerate(gt) if r.cls == cls_id]
                    t_idx = [i for i, r in enumerate(tr) if r.cls == cls_id]
                    sims.append(box_iou_matrix(gt_geom[g_idx], tr_geom[t_idx]))
            gt_ids, tr_ids, n_gt, n_tr = remap_ids(gt_ids, tr_ids)
            data_seq = {
                "num_timesteps": len(gt_ids),
                "num_gt_ids": n_gt,
                "num_tracker_ids": n_tr,
                "num_gt_dets": int(sum(len(x) for x in gt_ids)),
                "num_tracker_dets": int(sum(len(x) for x in tr_ids)),
                "gt_ids": gt_ids,
                "tracker_ids": tr_ids,
                "similarity_scores": sims,
            }
            seq_results[seq.name] = {
                "HOTA": HOTA.eval_sequence(data_seq),
                "CLEAR": CLEAR().eval_sequence(data_seq),
                "Identity": Identity().eval_sequence(data_seq),
            }

        hota_comb = HOTA.combine_sequences({s: v["HOTA"] for s, v in seq_results.items()})
        clear_comb = CLEAR.combine_sequences({s: v["CLEAR"] for s, v in seq_results.items()})
        id_comb = Identity.combine_sequences({s: v["Identity"] for s, v in seq_results.items()})
        per_class[class_name] = _merge_metrics(hota_comb, clear_comb, id_comb)
        for seq_name, values in seq_results.items():
            per_sequence.setdefault(seq_name, {})[class_name] = _merge_metrics(
                values["HOTA"], values["CLEAR"], values["Identity"]
            )

    hota_avg = HOTA.combine_classes_class_averaged(
        {str(model.names.get(int(c), c)): per_class[str(model.names.get(int(c), c))] for c in classes},
        ignore_empty_classes=True,
    )
    clear_avg = CLEAR.combine_classes_class_averaged(
        {str(model.names.get(int(c), c)): per_class[str(model.names.get(int(c), c))] for c in classes},
        ignore_empty_classes=True,
    )
    id_avg = Identity.combine_classes_class_averaged(
        {str(model.names.get(int(c), c)): per_class[str(model.names.get(int(c), c))] for c in classes},
        ignore_empty_classes=True,
    )
    hota_det = HOTA.combine_classes_det_averaged(per_class)
    clear_det = CLEAR.combine_classes_det_averaged(per_class)
    id_det = Identity.combine_classes_det_averaged(per_class)
    class_avg = _merge_metrics(hota_avg, clear_avg, id_avg)
    det_avg = _merge_metrics(hota_det, clear_det, id_det)

    _print_table(per_class, class_avg, det_avg)
    results_dict = {
        "metrics/HOTA": float(np.mean(class_avg["HOTA"])),
        "metrics/DetA": float(np.mean(class_avg["DetA"])),
        "metrics/AssA": float(np.mean(class_avg["AssA"])),
        "metrics/LocA": float(np.mean(class_avg["LocA"])),
        "metrics/MOTA": float(class_avg["MOTA"]),
        "metrics/MOTP": float(class_avg["MOTP"]),
        "metrics/IDF1": float(class_avg["IDF1"]),
        "metrics/IDP": float(class_avg["IDP"]),
        "metrics/IDR": float(class_avg["IDR"]),
        "fitness": float(np.mean(class_avg["HOTA"])),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable({"class_averaged": class_avg, "detection_averaged": det_avg,
                             "per_class": per_class, "per_sequence": per_sequence}), f,
                  ensure_ascii=False, indent=2)
    with (save_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        columns = ["sequence", "class", "HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDP", "IDR",
                   "CLR_FP", "CLR_FN", "IDSW"]
        writer.writerow(columns)
        for seq_name, classes_res in per_sequence.items():
            for class_name, res in classes_res.items():
                writer.writerow([
                    seq_name, class_name, *[
                        float(np.mean(res[col])) if col in ("HOTA", "DetA", "AssA", "LocA") else float(res[col])
                        for col in columns[2:]
                    ]
                ])
    return TrackValResults(
        hota=float(np.mean(hota_avg["HOTA"])),
        mota=float(clear_avg["MOTA"]),
        idf1=float(id_avg["IDF1"]),
        results_dict=results_dict,
        per_class=per_class,
        per_sequence=per_sequence,
        save_dir=save_dir,
    )
