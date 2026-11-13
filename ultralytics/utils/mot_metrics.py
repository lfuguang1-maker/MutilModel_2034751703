# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""HOTA / CLEAR / Identity metrics for offline multi-object tracking.

The metric definitions follow TrackEval (MIT License, https://github.com/JonathonLuiten/TrackEval).
Input ``data`` dictionaries match TrackEval's ``eval_sequence`` schema:
``num_timesteps``, ``num_gt_ids``, ``num_tracker_ids``, ``num_gt_dets``,
``num_tracker_dets``, ``gt_ids``, ``tracker_ids`` and ``similarity_scores``.
"""

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of ``[x1, y1, x2, y2]`` boxes."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, np.finfo(np.float64).eps)


def polygon_iou_matrix(polygons_a: np.ndarray, polygons_b: np.ndarray) -> np.ndarray:
    """Compute exact IoU between convex quadrilaterals shaped ``[N, 4, 2]``."""
    polygons_a = np.asarray(polygons_a, dtype=np.float32).reshape(-1, 4, 2)
    polygons_b = np.asarray(polygons_b, dtype=np.float32).reshape(-1, 4, 2)
    n_a, n_b = len(polygons_a), len(polygons_b)
    iou = np.zeros((n_a, n_b), dtype=np.float64)
    if n_a == 0 or n_b == 0:
        return iou

    bbox_a = np.stack([polygons_a[..., 0].min(1), polygons_a[..., 1].min(1),
                       polygons_a[..., 0].max(1), polygons_a[..., 1].max(1)], axis=1)
    bbox_b = np.stack([polygons_b[..., 0].min(1), polygons_b[..., 1].min(1),
                       polygons_b[..., 0].max(1), polygons_b[..., 1].max(1)], axis=1)
    bbox_iou = box_iou_matrix(bbox_a, bbox_b)
    area_a = np.array([abs(cv2.contourArea(p)) for p in polygons_a], dtype=np.float64)
    area_b = np.array([abs(cv2.contourArea(p)) for p in polygons_b], dtype=np.float64)

    for i, j in zip(*np.nonzero(bbox_iou > 0)):
        inter, _ = cv2.intersectConvexConvex(polygons_a[i], polygons_b[j])
        union = area_a[i] + area_b[j] - inter
        iou[i, j] = float(np.clip(inter / union, 0.0, 1.0)) if union > 0 else 0.0
    return iou


def apply_ignore_regions(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    ignore_mask: np.ndarray,
    similarity_fn,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove predictions matched to ignore GT and drop ignore GT.

    Returns ``(keep_gt_mask, keep_pred_indices)``.
    """
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    keep_pred = np.arange(n_pred, dtype=int)
    if n_gt == 0 or n_pred == 0 or not np.any(ignore_mask):
        return ~np.asarray(ignore_mask, dtype=bool), keep_pred

    similarity = similarity_fn(gt_boxes, pred_boxes)
    rows, cols = linear_sum_assignment(-similarity)
    matched = similarity[rows, cols] >= threshold
    ignored_pred = {int(cols[k]) for k in np.flatnonzero(matched) if ignore_mask[int(rows[k])]}
    keep_pred = np.array([i for i in range(n_pred) if i not in ignored_pred], dtype=int)
    return ~np.asarray(ignore_mask, dtype=bool), keep_pred


def remap_ids(
    gt_ids: Sequence[np.ndarray], tracker_ids: Sequence[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Remap arbitrary positive ids to contiguous indices starting at zero."""
    gt_mapping = {}
    tr_mapping = {}
    new_gt, new_tr = [], []
    for g, t in zip(gt_ids, tracker_ids):
        g = np.asarray(g)
        t = np.asarray(t)
        new_gt.append(np.array([gt_mapping.setdefault(int(v), len(gt_mapping)) for v in g], dtype=int))
        new_tr.append(np.array([tr_mapping.setdefault(int(v), len(tr_mapping)) for v in t], dtype=int))
    return new_gt, new_tr, len(gt_mapping), len(tr_mapping)


def _combine_sum(all_res: Dict[str, Dict], field: str):
    return sum(res[field] for res in all_res.values())


def _combine_weighted_av(all_res: Dict[str, Dict], field: str, combined: Dict, weight_field: str):
    return sum(res[field] * res[weight_field] for res in all_res.values()) / np.maximum(1.0, combined[weight_field])


class HOTA:
    """HOTA metric (TrackEval-compatible fields)."""

    array_labels = np.arange(0.05, 0.99, 0.05)
    float_array_fields = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"]
    integer_array_fields = ["HOTA_TP", "HOTA_FN", "HOTA_FP"]
    float_fields = ["HOTA(0)", "LocA(0)", "HOTALocA(0)"]

    @classmethod
    def eval_sequence(cls, data: Dict) -> Dict:
        res = {
            field: np.zeros(len(cls.array_labels), dtype=float)
            for field in cls.float_array_fields + cls.integer_array_fields
        }
        for field in cls.float_fields:
            res[field] = 0.0
        if data["num_tracker_dets"] == 0:
            res["HOTA_FN"] = data["num_gt_dets"] * np.ones(len(cls.array_labels))
            res["LocA"] = np.ones(len(cls.array_labels))
            res["LocA(0)"] = 1.0
            return res
        if data["num_gt_dets"] == 0:
            res["HOTA_FP"] = data["num_tracker_dets"] * np.ones(len(cls.array_labels))
            res["LocA"] = np.ones(len(cls.array_labels))
            res["LocA(0)"] = 1.0
            return res

        potential_matches_count = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
        gt_id_count = np.zeros((data["num_gt_ids"], 1))
        tracker_id_count = np.zeros((1, data["num_tracker_ids"]))
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            similarity = data["similarity_scores"][t]
            denom = similarity.sum(0)[None, :] + similarity.sum(1)[:, None] - similarity
            sim_iou = np.zeros_like(similarity)
            mask = denom > np.finfo(float).eps
            sim_iou[mask] = similarity[mask] / denom[mask]
            potential_matches_count[gt_ids_t[:, None], tracker_ids_t[None, :]] += sim_iou
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        global_alignment_score = potential_matches_count / np.maximum(
            1, gt_id_count + tracker_id_count - potential_matches_count
        )
        matches_counts = [np.zeros_like(potential_matches_count) for _ in cls.array_labels]

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            if len(gt_ids_t) == 0:
                res["HOTA_FP"] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res["HOTA_FN"] += len(gt_ids_t)
                continue
            similarity = data["similarity_scores"][t]
            score_mat = global_alignment_score[gt_ids_t[:, None], tracker_ids_t[None, :]] * similarity
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            for a, alpha in enumerate(cls.array_labels):
                actually_matched = similarity[match_rows, match_cols] >= alpha - np.finfo(float).eps
                rows, cols = match_rows[actually_matched], match_cols[actually_matched]
                num_matches = len(rows)
                res["HOTA_TP"][a] += num_matches
                res["HOTA_FN"][a] += len(gt_ids_t) - num_matches
                res["HOTA_FP"][a] += len(tracker_ids_t) - num_matches
                if num_matches:
                    res["LocA"][a] += float(similarity[rows, cols].sum())
                    matches_counts[a][gt_ids_t[rows], tracker_ids_t[cols]] += 1

        for a in range(len(cls.array_labels)):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res["AssA"][a] = np.sum(matches_count * ass_a) / np.maximum(1, res["HOTA_TP"][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res["AssRe"][a] = np.sum(matches_count * ass_re) / np.maximum(1, res["HOTA_TP"][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res["AssPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res["HOTA_TP"][a])

        res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
        return cls._compute_final_fields(res)

    @classmethod
    def combine_sequences(cls, all_res: Dict[str, Dict]) -> Dict:
        res = {field: _combine_sum(all_res, field) for field in cls.integer_array_fields}
        for field in ("AssRe", "AssPr", "AssA"):
            res[field] = _combine_weighted_av(all_res, field, res, "HOTA_TP")
        loca_sum = sum(v["LocA"] * v["HOTA_TP"] for v in all_res.values())
        res["LocA"] = np.maximum(1e-10, loca_sum) / np.maximum(1e-10, res["HOTA_TP"])
        return cls._compute_final_fields(res)

    @classmethod
    def combine_classes_class_averaged(cls, all_res: Dict[str, Dict], ignore_empty_classes: bool = False) -> Dict:
        selected = {
            k: v for k, v in all_res.items()
            if not ignore_empty_classes or (v["HOTA_TP"] + v["HOTA_FN"] + v["HOTA_FP"] > np.finfo(float).eps).any()
        }
        res = {field: _combine_sum(selected, field) for field in cls.integer_array_fields}
        for field in cls.float_fields + cls.float_array_fields:
            res[field] = np.mean([v[field] for v in selected.values()], axis=0) if selected else 0.0
        return res

    @classmethod
    def combine_classes_det_averaged(cls, all_res: Dict[str, Dict]) -> Dict:
        return cls.combine_sequences(all_res)

    @classmethod
    def _compute_final_fields(cls, res: Dict) -> Dict:
        res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
        res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
        res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
        res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
        res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
        res["HOTA(0)"] = res["HOTA"][0]
        res["LocA(0)"] = res["LocA"][0]
        res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
        return res


class CLEAR:
    """CLEAR MOT metric (TrackEval-compatible fields)."""

    integer_fields = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "CLR_Frames"]
    float_fields = ["MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR",
                    "sMOTA", "CLR_F1", "FP_per_frame", "MOTAL", "MOTP_sum"]

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def eval_sequence(self, data: Dict) -> Dict:
        res = {field: 0.0 for field in self.float_fields + self.integer_fields}
        if data["num_tracker_dets"] == 0:
            res["CLR_FN"] = data["num_gt_dets"]
            res["ML"] = data["num_gt_ids"]
            res["MLR"] = 1.0
            return res
        if data["num_gt_dets"] == 0:
            res["CLR_FP"] = data["num_tracker_dets"]
            res["MLR"] = 1.0
            return res

        gt_id_count = np.zeros(data["num_gt_ids"])
        gt_matched_count = np.zeros(data["num_gt_ids"])
        gt_frag_count = np.zeros(data["num_gt_ids"])
        prev_tracker_id = np.nan * np.zeros(data["num_gt_ids"])
        prev_timestep_tracker_id = np.nan * np.zeros(data["num_gt_ids"])

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            if len(gt_ids_t) == 0:
                res["CLR_FP"] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res["CLR_FN"] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            similarity = data["similarity_scores"][t]
            score_mat = (tracker_ids_t[None, :] == prev_timestep_tracker_id[gt_ids_t[:, None]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo(float).eps] = 0
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            matched = score_mat[match_rows, match_cols] > np.finfo(float).eps
            match_rows, match_cols = match_rows[matched], match_cols[matched]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]
            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            switched = ~np.isnan(prev_matched_tracker_ids) & (matched_tracker_ids != prev_matched_tracker_ids)
            res["IDSW"] += np.sum(switched)

            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
            currently_tracked = ~np.isnan(prev_timestep_tracker_id)
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            num_matches = len(matched_gt_ids)
            res["CLR_TP"] += num_matches
            res["CLR_FN"] += len(gt_ids_t) - num_matches
            res["CLR_FP"] += len(tracker_ids_t) - num_matches
            if num_matches:
                res["MOTP_sum"] += float(similarity[match_rows, match_cols].sum())

        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res["MT"] = int(np.sum(tracked_ratio > 0.8))
        res["PT"] = int(np.sum(tracked_ratio >= 0.2) - res["MT"])
        res["ML"] = int(data["num_gt_ids"] - res["MT"] - res["PT"])
        res["Frag"] = int(np.sum(gt_frag_count[gt_frag_count > 0] - 1))
        res["CLR_Frames"] = data["num_timesteps"]
        res["MOTP"] = res["MOTP_sum"] / max(1.0, res["CLR_TP"])
        return self._compute_final_fields(res)

    @classmethod
    def combine_sequences(cls, all_res: Dict[str, Dict]) -> Dict:
        res = {field: _combine_sum(all_res, field) for field in cls.integer_fields + ["MOTP_sum"]}
        return cls._compute_final_fields(res)

    @classmethod
    def combine_classes_det_averaged(cls, all_res: Dict[str, Dict]) -> Dict:
        return cls.combine_sequences(all_res)

    @classmethod
    def combine_classes_class_averaged(cls, all_res: Dict[str, Dict], ignore_empty_classes: bool = False) -> Dict:
        selected = {
            k: v for k, v in all_res.items()
            if not ignore_empty_classes or v["CLR_TP"] + v["CLR_FN"] + v["CLR_FP"] > 0
        }
        res = {field: _combine_sum(selected, field) for field in cls.integer_fields}
        for field in cls.float_fields:
            res[field] = float(np.mean([v[field] for v in selected.values()])) if selected else 0.0
        return res

    @staticmethod
    def _compute_final_fields(res: Dict) -> Dict:
        num_gt_ids = res["MT"] + res["ML"] + res["PT"]
        res["MTR"] = res["MT"] / max(1.0, num_gt_ids)
        res["MLR"] = res["ML"] / max(1.0, num_gt_ids)
        res["PTR"] = res["PT"] / max(1.0, num_gt_ids)
        res["CLR_Re"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["CLR_Pr"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FP"])
        res["MODA"] = (res["CLR_TP"] - res["CLR_FP"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["MOTP"] = res["MOTP_sum"] / max(1.0, res["CLR_TP"])
        res["sMOTA"] = (res["MOTP_sum"] - res["CLR_FP"] - res["IDSW"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["CLR_F1"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + 0.5 * res["CLR_FN"] + 0.5 * res["CLR_FP"])
        res["FP_per_frame"] = res["CLR_FP"] / max(1.0, res["CLR_Frames"])
        safe_log_idsw = np.log10(res["IDSW"]) if res["IDSW"] > 0 else res["IDSW"]
        res["MOTAL"] = (res["CLR_TP"] - res["CLR_FP"] - safe_log_idsw) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        return res


class Identity:
    """Identity metric (TrackEval-compatible fields)."""

    integer_fields = ["IDTP", "IDFN", "IDFP"]
    float_fields = ["IDF1", "IDR", "IDP"]

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def eval_sequence(self, data: Dict) -> Dict:
        res = {field: 0.0 for field in self.float_fields + self.integer_fields}
        if data["num_tracker_dets"] == 0:
            res["IDFN"] = data["num_gt_dets"]
            return res
        if data["num_gt_dets"] == 0:
            res["IDFP"] = data["num_tracker_dets"]
            return res

        potential_matches_count = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
        gt_id_count = np.zeros(data["num_gt_ids"])
        tracker_id_count = np.zeros(data["num_tracker_ids"])
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            matches_mask = data["similarity_scores"][t] >= self.threshold
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1

        num_gt_ids, num_tracker_ids = data["num_gt_ids"], data["num_tracker_ids"]
        fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        fn_mat = np.zeros_like(fp_mat)
        fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
        fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
        for gt_id in range(num_gt_ids):
            fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
            fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
        for tracker_id in range(num_tracker_ids):
            fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
            fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
        fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
        fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)
        res["IDFN"] = int(fn_mat[match_rows, match_cols].sum())
        res["IDFP"] = int(fp_mat[match_rows, match_cols].sum())
        res["IDTP"] = int(gt_id_count.sum() - res["IDFN"])
        return self._compute_final_fields(res)

    @classmethod
    def combine_sequences(cls, all_res: Dict[str, Dict]) -> Dict:
        res = {field: _combine_sum(all_res, field) for field in cls.integer_fields}
        return cls._compute_final_fields(res)

    @classmethod
    def combine_classes_det_averaged(cls, all_res: Dict[str, Dict]) -> Dict:
        return cls.combine_sequences(all_res)

    @classmethod
    def combine_classes_class_averaged(cls, all_res: Dict[str, Dict], ignore_empty_classes: bool = False) -> Dict:
        selected = {
            k: v for k, v in all_res.items()
            if not ignore_empty_classes or v["IDTP"] + v["IDFN"] + v["IDFP"] > np.finfo(float).eps
        }
        res = {field: _combine_sum(selected, field) for field in cls.integer_fields}
        for field in cls.float_fields:
            res[field] = float(np.mean([v[field] for v in selected.values()])) if selected else 0.0
        return res

    @staticmethod
    def _compute_final_fields(res: Dict) -> Dict:
        res["IDR"] = res["IDTP"] / max(1.0, res["IDTP"] + res["IDFN"])
        res["IDP"] = res["IDTP"] / max(1.0, res["IDTP"] + res["IDFP"])
        res["IDF1"] = res["IDTP"] / max(1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"])
        return res
