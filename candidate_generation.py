# candidate_generation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np


# Config
@dataclass(frozen=True)
class CandidateGenConfig:
    # -------- diff_n --------
    eps: float = 1e-3
    use_mad: bool = True
    mad_k: float = 8.0
    q_high: float = 0.985   # when MAD disabled

    # -------- morphology (critical) --------
    morph_scale: float = 0.004   # kernel = scale * min(H, W)
    min_area_frac: float = 0.00005

    # -------- box post --------
    bbox_pad_frac: float = 0.25
    max_boxes: int = 50

    # nested suppression
    nested_iou: float = 0.80
    nested_area_ratio: float = 0.25

    # -------- background (KNN) --------
    enable_bg: bool = True
    bg_history: int = 300
    bg_dist2: float = 400.0
    bg_thresh: int = 200


@dataclass
class CandidateGenResult:
    mask: np.ndarray
    boxes: List[Tuple[int, int, int, int]]
    debug: Dict[str, Any]


# Generator
class MotionCandidateGenerator:
    def __init__(self, cfg: CandidateGenConfig):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorKNN(
            history=cfg.bg_history,
            dist2Threshold=cfg.bg_dist2,
            detectShadows=False,
        )

    def reset(self):
        self.__init__(self.cfg)


# Utils
def _odd(k: int) -> int:
    return int(k + (k % 2 == 0))


def _kernel(cfg: CandidateGenConfig, H: int, W: int) -> int:
    return _odd(max(3, int(round(cfg.morph_scale * min(H, W)))))


def _mad_threshold(x: np.ndarray, valid: np.ndarray, k: float) -> float:
    v = x[valid]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    return med + k * mad


def _nested_suppress(
    boxes: List[Tuple[int, int, int, int]],
    iou_thr: float,
    area_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    areas = np.array([w * h for (_, _, w, h) in boxes])
    order = np.argsort(-areas)
    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(order)):
        if not keep[order[i]]:
            continue
        xi, yi, wi, hi = boxes[order[i]]
        ai = wi * hi
        for j in range(i + 1, len(order)):
            if not keep[order[j]]:
                continue
            xj, yj, wj, hj = boxes[order[j]]
            aj = wj * hj
            if aj > area_ratio * ai:
                continue

            xx1 = max(xi, xj)
            yy1 = max(yi, yj)
            xx2 = min(xi + wi, xj + wj)
            yy2 = min(yi + hi, yj + hj)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if inter / aj >= iou_thr:
                keep[order[j]] = False

    return [b for b, k in zip(boxes, keep) if k]


# Main function
def generate_motion_candidates(
    curr_I_norm: np.ndarray,
    prev_I_norm_aligned: np.ndarray,
    valid_mask: np.ndarray,
    gen: MotionCandidateGenerator,
) -> CandidateGenResult:
    """
    New Step3 (spec_mask-free, fill-free, I_norm friendly)
    """

    cfg = gen.cfg
    H, W = curr_I_norm.shape
    valid = valid_mask > 0

    debug: Dict[str, Any] = {}

    # 1. diff_n (seed)
    a = curr_I_norm.astype(np.float32)
    b = prev_I_norm_aligned.astype(np.float32)
    diff_n = np.abs(a - b) / (a + b + cfg.eps)
    diff_n[~valid] = 0.0

    # threshold
    if cfg.use_mad:
        thr = _mad_threshold(diff_n, valid, cfg.mad_k)
        thr_type = "MAD"
    else:
        thr = np.quantile(diff_n[valid], cfg.q_high)
        thr_type = "quantile"

    seed = diff_n > thr
    debug["diff"] = {
        "thr": float(thr),
        "type": thr_type,
        "fg_ratio": float(np.mean(seed[valid])),
    }

    # 2. Morphology
    k = _kernel(cfg, H, W)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = seed.astype(np.uint8) * 255
    mask = cv2.dilate(mask, ker, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)
    mask = cv2.erode(mask, ker, iterations=1)
    mask[~valid] = 0

    debug["morph"] = {"kernel": k}

    # 3. Background OR (KNN)
    if cfg.enable_bg:
        bg = gen.bg.apply(curr_I_norm, learningRate=0)
        _, bg = cv2.threshold(bg, cfg.bg_thresh, 255, cv2.THRESH_BINARY)
        bg[~valid] = 0
        mask = cv2.bitwise_or(mask, bg)
        debug["bg"] = True
    else:
        debug["bg"] = False

    # 4. CC -> boxes
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = int(cfg.min_area_frac * H * W)
    boxes: List[Tuple[int, int, int, int]] = []

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        px = int(round(w * cfg.bbox_pad_frac))
        py = int(round(h * cfg.bbox_pad_frac))
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(W, x + w + px)
        y1 = min(H, y + h + py)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    debug["cc"] = {"raw_boxes": len(boxes)}

    # 5. Parent-child suppression
    boxes = _nested_suppress(
        boxes,
        cfg.nested_iou,
        cfg.nested_area_ratio,
    )

    boxes = boxes[: cfg.max_boxes]
    debug["final_boxes"] = len(boxes)

    return CandidateGenResult(
        mask=mask,
        boxes=boxes,
        debug=debug,
    )

