# multi_tracker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


BBox = Tuple[int, int, int, int]


def bbox_xywh_to_cxcywh(b: BBox) -> np.ndarray:
    x, y, w, h = b
    cx = x + w / 2.0
    cy = y + h / 2.0
    return np.array([cx, cy, float(w), float(h)], dtype=np.float32)


def cxcywh_to_bbox(v: np.ndarray) -> BBox:
    cx, cy, w, h = map(float, v[:4])
    x = cx - w / 2.0
    y = cy - h / 2.0
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple fallback if scipy not available: greedy row-wise assignment.
    Returns row_idx, col_idx.
    """
    pairs = []
    used_cols = set()
    for r in range(cost.shape[0]):
        c = int(np.argmin(cost[r]))
        if c not in used_cols:
            pairs.append((r, c))
            used_cols.add(c)
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    rr = np.array([p[0] for p in pairs], dtype=int)
    cc = np.array([p[1] for p in pairs], dtype=int)
    return rr, cc


@dataclass
class KalmanParams:
    dt: float = 1.0
    # noise scales (tune!)
    process_noise_pos: float = 1e-2   # affects smoothness vs responsiveness
    process_noise_vel: float = 1e-1
    meas_noise_pos: float = 1e-1
    meas_noise_size: float = 1e-1


def create_kalman_filter(params: KalmanParams) -> cv2.KalmanFilter:
    """
    State: [cx, cy, w, h, vx, vy, vw, vh]^T  (8)
    Meas : [cx, cy, w, h]^T                  (4)
    """
    kf = cv2.KalmanFilter(8, 4)

    dt = params.dt
    # transition matrix
    F = np.eye(8, dtype=np.float32)
    F[0, 4] = dt
    F[1, 5] = dt
    F[2, 6] = dt
    F[3, 7] = dt
    kf.transitionMatrix = F

    # measurement matrix
    H = np.zeros((4, 8), dtype=np.float32)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0
    kf.measurementMatrix = H

    # process noise covariance Q
    Q = np.eye(8, dtype=np.float32)
    Q[0, 0] = params.process_noise_pos
    Q[1, 1] = params.process_noise_pos
    Q[2, 2] = params.process_noise_pos
    Q[3, 3] = params.process_noise_pos
    Q[4, 4] = params.process_noise_vel
    Q[5, 5] = params.process_noise_vel
    Q[6, 6] = params.process_noise_vel
    Q[7, 7] = params.process_noise_vel
    kf.processNoiseCov = Q

    # measurement noise covariance R
    R = np.eye(4, dtype=np.float32)
    R[0, 0] = params.meas_noise_pos
    R[1, 1] = params.meas_noise_pos
    R[2, 2] = params.meas_noise_size
    R[3, 3] = params.meas_noise_size
    kf.measurementNoiseCov = R

    # initial state covariance P
    kf.errorCovPost = np.eye(8, dtype=np.float32)

    return kf


@dataclass
class Track:
    track_id: int
    kf: cv2.KalmanFilter
    bbox: BBox
    points: Optional[np.ndarray] = None  # Step5 points
    age: int = 0
    hits: int = 0
    miss: int = 0
    confirmed: bool = False

    def predict(self) -> BBox:
        pred = self.kf.predict()  # 8x1
        v = pred.reshape(-1)
        self.bbox = cxcywh_to_bbox(v[:4])
        self.age += 1
        return self.bbox

    def update(self, det_bbox: BBox):
        z = bbox_xywh_to_cxcywh(det_bbox).reshape(4, 1)
        self.kf.correct(z)
        state = self.kf.statePost.reshape(-1)
        self.bbox = cxcywh_to_bbox(state[:4])
        self.hits += 1
        self.miss = 0
        if self.hits >= 2:  # confirm after 2 hits (typical)
            self.confirmed = True


@dataclass
class MOTParams:
    kalman: KalmanParams = KalmanParams()

    # association
    iou_weight: float = 0.7
    center_weight: float = 0.3
    max_center_dist: float = 150.0     # gating in pixels (tune to resolution)
    min_iou: float = 0.05              # gating lower bound

    # track management
    max_miss: int = 10                 # delete track if missed too long
    min_confirm_hits: int = 2          # confirm threshold (also in Track.update)
    init_with_detection: bool = True


def bbox_center(b: BBox) -> Tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def center_dist(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


def build_cost_matrix(tracks: List[Track], dets: List[BBox], params: MOTParams) -> np.ndarray:
    """
    cost = iou_weight*(1-IOU) + center_weight*(dist/max_center_dist)
    with gating: if IOU < min_iou and dist > max_center_dist => set high cost
    """
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)), dtype=np.float32)

    cost = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    big = 1e6

    for i, tr in enumerate(tracks):
        for j, d in enumerate(dets):
            iou_ = iou(tr.bbox, d)
            dist_ = center_dist(tr.bbox, d)
            if (iou_ < params.min_iou) and (dist_ > params.max_center_dist):
                cost[i, j] = big
            else:
                c_iou = (1.0 - iou_)
                c_dist = min(1.0, dist_ / max(1e-6, params.max_center_dist))
                cost[i, j] = params.iou_weight * c_iou + params.center_weight * c_dist
    return cost


class MultiObjectTracker:
    def __init__(self, params: MOTParams = MOTParams()):
        self.params = params
        self.tracks: List[Track] = []
        self._next_id = 1

    def _new_track(self, det: BBox) -> Track:
        kf = create_kalman_filter(self.params.kalman)
        # init statePost from detection
        z = bbox_xywh_to_cxcywh(det)
        kf.statePost = np.array([[z[0]], [z[1]], [z[2]], [z[3]],
                                 [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        tr = Track(track_id=self._next_id, kf=kf, bbox=det)
        self._next_id += 1
        return tr

    def predict(self):
        for tr in self.tracks:
            tr.predict()

    def update(self, detections: List[BBox]) -> Dict[str, Any]:
        """
        Main Step6 update:
          1) predict all tracks
          2) associate with detections
          3) correct matched tracks, mark misses, create new tracks, delete dead tracks
        """
        dbg: Dict[str, Any] = {"num_tracks_before": len(self.tracks), "num_dets": len(detections)}

        # 1) predict
        self.predict()

        # 2) association
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(self._new_track(det))
            dbg["created"] = len(detections)
            dbg["num_tracks_after"] = len(self.tracks)
            return dbg

        if len(detections) == 0:
            # no detections -> all miss
            for tr in self.tracks:
                tr.miss += 1
            self.tracks = [t for t in self.tracks if t.miss <= self.params.max_miss]
            dbg["all_missed"] = True
            dbg["num_tracks_after"] = len(self.tracks)
            return dbg

        cost = build_cost_matrix(self.tracks, detections, self.params)

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = greedy_assignment(cost)

        matched_t = set()
        matched_d = set()
        matches = []

        big = 1e6
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[r, c] >= big:
                continue
            matched_t.add(r)
            matched_d.add(c)
            matches.append((r, c))

        # 3) update matched
        for r, c in matches:
            self.tracks[r].update(detections[c])

        # unmatched tracks -> miss++
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.miss += 1

        # unmatched detections -> new tracks
        created = 0
        for j, det in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(self._new_track(det))
                created += 1

        # delete dead tracks
        self.tracks = [t for t in self.tracks if t.miss <= self.params.max_miss]

        dbg["matches"] = len(matches)
        dbg["created"] = created
        dbg["num_tracks_after"] = len(self.tracks)

        return dbg

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]
