# point_tracker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2


def bbox_to_roi(bbox: Tuple[int, int, int, int], H: int, W: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def clamp_bbox(x: float, y: float, w: float, h: float, H: int, W: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(round(x)), W - 1))
    y = max(0, min(int(round(y)), H - 1))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return x, y, w, h


def apply_valid_mask(img: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    if valid_mask is None:
        return img
    m = valid_mask.astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=m)


def detect_points_in_bbox(
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    valid_mask: Optional[np.ndarray] = None,
    max_corners: int = 150,
    quality_level: float = 0.01,
    min_distance: int = 7,
    block_size: int = 7,
) -> Optional[np.ndarray]:
    """
    Shi–Tomasi corners inside bbox.
    Return points as Nx1x2 float32 for LK.
    """
    H, W = gray.shape[:2]
    x, y, w, h = bbox_to_roi(bbox, H, W)
    roi = gray[y:y+h, x:x+w]

    if valid_mask is not None:
        vm = valid_mask[y:y+h, x:x+w].astype(np.uint8)
    else:
        vm = None

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=False
    )
    if corners is None:
        return None

    # shift ROI coords -> full image coords
    corners[:, 0, 0] += x
    corners[:, 0, 1] += y

    # apply valid mask filtering if provided
    if valid_mask is not None:
        pts = corners[:, 0, :].astype(int)
        keep = (valid_mask[pts[:, 1], pts[:, 0]] > 0)
        corners = corners[keep]
        if len(corners) == 0:
            return None

    return corners.astype(np.float32)


def lk_track_points(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    p0: np.ndarray,
    win_size: int = 21,
    max_level: int = 3,
    criteria_iters: int = 20,
    criteria_eps: float = 0.03,
    min_eig_threshold: float = 1e-4,
    fb_check: bool = True,
    fb_thresh: float = 1.5,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Track points p0 from prev->curr using pyramidal LK.
    Optionally do forward-backward check for robustness.
    """
    dbg: Dict[str, Any] = {"p0": int(len(p0)), "kept": 0, "fb_check": fb_check}

    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, criteria_iters, criteria_eps)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=lk_criteria,
        flags=0,
        minEigThreshold=min_eig_threshold
    )
    if p1 is None or st is None:
        dbg["reason"] = "lk_forward_failed"
        return None, dbg

    st = st.reshape(-1).astype(bool)
    p0f = p0[st]
    p1f = p1[st]

    if len(p1f) == 0:
        dbg["reason"] = "no_forward_points"
        return None, dbg

    keep = np.ones((len(p1f),), dtype=bool)

    if fb_check:
        # Backward tracking: curr->prev
        p0b, stb, errb = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, p1f, None,
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=lk_criteria,
            flags=0,
            minEigThreshold=min_eig_threshold
        )
        if p0b is None or stb is None:
            dbg["reason"] = "lk_backward_failed"
            return None, dbg

        stb = stb.reshape(-1).astype(bool)
        keep = keep & stb
        p0f2 = p0f[keep]
        p1f2 = p1f[keep]
        p0b2 = p0b[keep]

        if len(p1f2) == 0:
            dbg["reason"] = "no_fb_points"
            return None, dbg

        fb_err = np.linalg.norm((p0b2 - p0f2).reshape(-1, 2), axis=1)
        keep2 = fb_err <= fb_thresh
        p1k = p1f2[keep2]
    else:
        p1k = p1f

    if p1k is None or len(p1k) == 0:
        dbg["reason"] = "no_points_after_filter"
        return None, dbg

    dbg["kept"] = int(len(p1k))
    return p1k.astype(np.float32), dbg


def bbox_from_points(
    pts: np.ndarray,
    H: int,
    W: int,
    padding: int = 4,
    min_wh: int = 10
) -> Tuple[int, int, int, int]:
    """
    Compute bbox from tracked points (Nx1x2).
    """
    xy = pts[:, 0, :]
    x_min = float(np.min(xy[:, 0])) - padding
    y_min = float(np.min(xy[:, 1])) - padding
    x_max = float(np.max(xy[:, 0])) + padding
    y_max = float(np.max(xy[:, 1])) + padding

    w = max(min_wh, x_max - x_min)
    h = max(min_wh, y_max - y_min)
    return clamp_bbox(x_min, y_min, w, h, H, W)


@dataclass
class Step5Params:
    # Shi–Tomasi
    max_corners: int = 150
    quality_level: float = 0.01
    min_distance: int = 7
    block_size: int = 7

    # LK
    win_size: int = 21
    max_level: int = 3
    criteria_iters: int = 20
    criteria_eps: float = 0.03
    min_eig_threshold: float = 1e-4
    fb_check: bool = True
    fb_thresh: float = 1.5

    # Re-init / bbox update
    min_points_to_track: int = 25
    bbox_padding: int = 4
    min_bbox_wh: int = 10


def step5_update_tracks_with_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    tracks: List[Any],  # expects track.bbox and track.points attributes
    valid_mask: Optional[np.ndarray] = None,
    params: Step5Params = Step5Params(),
) -> Dict[str, Any]:
    """
    Update each track:
      - if points exist: LK track points, update bbox from points
      - if too few points / LK fails: re-detect points in current bbox

    Mutates tracks in-place.
    Returns debug summary.
    """
    prev_gray = prev_gray.astype(np.uint8)
    curr_gray = curr_gray.astype(np.uint8)

    H, W = curr_gray.shape[:2]
    dbg_all: Dict[str, Any] = {"tracks": []}

    # Apply valid mask to images for more stable tracking if needed
    prev_use = apply_valid_mask(prev_gray, valid_mask)
    curr_use = apply_valid_mask(curr_gray, valid_mask)

    for tr in tracks:
        info = {"id": tr.track_id, "status": None}
        p0 = getattr(tr, "points", None)

        # If no points or too few, (re)detect in bbox on prev frame to track forward
        if p0 is None or len(p0) < params.min_points_to_track:
            p0 = detect_points_in_bbox(
                prev_use, tr.bbox, valid_mask=valid_mask,
                max_corners=params.max_corners,
                quality_level=params.quality_level,
                min_distance=params.min_distance,
                block_size=params.block_size
            )
            tr.points = p0
            info["status"] = "reinit_points" if p0 is not None else "no_points"
            dbg_all["tracks"].append(info)
            continue

        # LK tracking
        p1, lk_dbg = lk_track_points(
            prev_use, curr_use, p0,
            win_size=params.win_size,
            max_level=params.max_level,
            criteria_iters=params.criteria_iters,
            criteria_eps=params.criteria_eps,
            min_eig_threshold=params.min_eig_threshold,
            fb_check=params.fb_check,
            fb_thresh=params.fb_thresh
        )
        info.update({"lk": lk_dbg})

        if p1 is None or len(p1) < params.min_points_to_track:
            # tracking unstable -> reinit points in current bbox (using current frame)
            p_new = detect_points_in_bbox(
                curr_use, tr.bbox, valid_mask=valid_mask,
                max_corners=params.max_corners,
                quality_level=params.quality_level,
                min_distance=params.min_distance,
                block_size=params.block_size
            )
            tr.points = p_new
            info["status"] = "lk_failed_reinit" if p_new is not None else "lk_failed_no_points"
            dbg_all["tracks"].append(info)
            continue

        # update bbox from tracked points
        tr.points = p1
        tr.bbox = bbox_from_points(
            p1, H, W,
            padding=params.bbox_padding,
            min_wh=params.min_bbox_wh
        )
        info["status"] = "ok"
        info["bbox"] = tr.bbox
        dbg_all["tracks"].append(info)

    return dbg_all
