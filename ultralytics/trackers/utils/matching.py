# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # optional dependency for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    lap = None  # fall back to scipy.optimize.linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """
    Perform linear assignment using lap.lapjv when available, otherwise scipy.

    The scipy fallback solves the same extended assignment problem as
    ``lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)``: the cost matrix is padded to
    ``(N + M) x (N + M)`` with ``thresh / 2`` in the off-diagonal blocks and zeros in the dummy block.
    Assignments to dummy rows or columns mean "unmatched". For large matrices the problem is split into
    connected components of the bipartite graph of pairs with ``cost < thresh``, which does not change the optimum.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment when available. If False or lap is not installed,
            scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (np.ndarray): Matched indices of shape (K, 2), where K is the number of matches.
        unmatched_a (np.ndarray): Array of unmatched indices from the first set, with shape (L,).
        unmatched_b (np.ndarray): Array of unmatched indices from the second set, with shape (M,).

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    if use_lap and lap is not None:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = np.asarray([[ix, mx] for ix, mx in enumerate(x) if mx >= 0], dtype=int).reshape(-1, 2)
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        _warn_scipy_fallback()
        matches, unmatched_a, unmatched_b = _scipy_linear_assignment(cost_matrix, thresh)

    return matches, unmatched_a, unmatched_b


_SCIPY_SPLIT_MIN_SIZE = 10_000  # cost-matrix entries above which connected-component splitting is used
_scipy_fallback_warned = False


def _warn_scipy_fallback():
    """Print a one-time hint about the scipy linear-assignment fallback."""
    global _scipy_fallback_warned
    if not _scipy_fallback_warned:
        hint = " Install 'lap>=0.5.12' for faster tracking." if lap is None else ""
        LOGGER.info(f"Using SciPy for linear assignment (results match lap).{hint}")
        _scipy_fallback_warned = True


def _scipy_linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Solve the extended assignment problem with scipy, splitting large problems into components."""
    n_rows, n_cols = cost_matrix.shape
    if n_rows * n_cols < _SCIPY_SPLIT_MIN_SIZE:
        return _solve_extended_assignment(cost_matrix, thresh)

    matches = []
    unmatched_a = np.ones(n_rows, dtype=bool)
    unmatched_b = np.ones(n_cols, dtype=bool)
    for rows, cols in _assignment_components(cost_matrix, thresh):
        sub_matches, _, _ = _solve_extended_assignment(cost_matrix[np.ix_(rows, cols)], thresh)
        if len(sub_matches):
            sub_rows, sub_cols = rows[sub_matches[:, 0]], cols[sub_matches[:, 1]]
            matches.extend(zip(sub_rows.tolist(), sub_cols.tolist()))
            unmatched_a[sub_rows] = False
            unmatched_b[sub_cols] = False
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    return matches, np.flatnonzero(unmatched_a), np.flatnonzero(unmatched_b)


def _solve_extended_assignment(cost_matrix: np.ndarray, thresh: float):
    """Solve one assignment problem using lap's extended-matrix formulation."""
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 1 and n_cols > 0:
        # A single row can only match its cheapest column; avoid a scipy call for this common component shape.
        j = int(np.argmin(cost_matrix[0]))
        if not np.isfinite(thresh) or cost_matrix[0, j] < thresh:
            return np.array([[0, j]], dtype=int), np.empty(0, dtype=int), np.delete(np.arange(n_cols), j)
        return np.empty((0, 2), dtype=int), np.array([0]), np.arange(n_cols)
    if n_cols == 1 and n_rows > 0:
        # A single column can only match its cheapest row.
        i = int(np.argmin(cost_matrix[:, 0]))
        if not np.isfinite(thresh) or cost_matrix[i, 0] < thresh:
            return np.array([[i, 0]], dtype=int), np.delete(np.arange(n_rows), i), np.empty(0, dtype=int)
        return np.empty((0, 2), dtype=int), np.arange(n_rows), np.array([0])
    if np.isfinite(thresh):
        # Same extension as lap.lapjv(..., extend_cost=True, cost_limit=thresh).
        size = n_rows + n_cols
        extended = np.full((size, size), thresh / 2, dtype=np.float64)
        extended[n_rows:, n_cols:] = 0
    else:
        # lap pads a rectangular matrix with zeros when no cost limit is given.
        size = max(n_rows, n_cols)
        extended = np.zeros((size, size), dtype=np.float64)
    extended[:n_rows, :n_cols] = cost_matrix

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(extended)
    matched = (row_ind < n_rows) & (col_ind < n_cols)
    rows, cols = row_ind[matched], col_ind[matched]
    if np.isfinite(thresh):
        # The extended problem is indifferent between matching at exactly `thresh` and leaving both sides
        # unmatched; keep the strict-threshold convention deterministic.
        keep = cost_matrix[rows, cols] < thresh
        rows, cols = rows[keep], cols[keep]

    matches = np.column_stack((rows, cols)).astype(int) if rows.size else np.empty((0, 2), dtype=int)
    used_a = np.zeros(n_rows, dtype=bool)
    used_b = np.zeros(n_cols, dtype=bool)
    used_a[rows] = True
    used_b[cols] = True
    return matches, np.flatnonzero(~used_a), np.flatnonzero(~used_b)


def _assignment_components(cost_matrix: np.ndarray, thresh: float):
    """Yield row and column index arrays for connected components of pairs with ``cost < thresh``."""
    n_rows, n_cols = cost_matrix.shape
    rows, cols = np.nonzero(cost_matrix < thresh)
    if rows.size == 0:
        return

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols + n_rows)), shape=(n_rows + n_cols, n_rows + n_cols)
    )
    _, labels = connected_components(graph, directed=False)
    row_labels, col_labels = labels[:n_rows], labels[n_rows:]
    for label in np.unique(row_labels[rows]):
        yield np.flatnonzero(row_labels == label), np.flatnonzero(col_labels == label)


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (List[STrack] | List[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (List[STrack] | List[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).

    Examples:
        Compute IoU distance between two sets of tracks
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (List[STrack]): List of tracks, where each track contains embedding features.
        detections (List[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks
            and M is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """
    Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (List[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
