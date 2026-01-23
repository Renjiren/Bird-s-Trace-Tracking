# candidate_generation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2


@dataclass
class CandidateGenResult:
    mask: np.ndarray                          # final binary foreground mask (uint8 0/255)
    boxes: List[Tuple[int, int, int, int]]    # list of (x, y, w, h)
    debug: Dict[str, Any]


def create_bg_subtractor_mog2(
    history: int = 300,
    var_threshold: float = 16.0,
    detect_shadows: bool = True,
) -> cv2.BackgroundSubtractor:
    """
    Create a persistent background subtractor.
    Must be created ONCE and reused across frames.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )


def _ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _apply_mask(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """mask: uint8 (255 valid, 0 ignore)."""
    if mask is None:
        return img
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=mask)


def _binary_cleanup(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    """
    Morphological cleanup: OPEN removes small noise; CLOSE fills small holes.
    """
    out = mask
    if k_open and k_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    if k_close and k_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


def generate_motion_candidates_step3(
    curr_gray: np.ndarray,
    prev_aligned: np.ndarray,
    bg_subtractor: Optional[cv2.BackgroundSubtractor] = None,
    valid_mask: Optional[np.ndarray] = None,     # e.g., subtitle-masked valid pixels
    # --- A) frame-diff branch
    diff_blur_ksize: int = 5,                    # smooth diff to suppress tiny glitter
    diff_thresh: int = 25,                       # threshold for absdiff
    # --- B) background subtraction branch
    bg_learning_rate: float = -1.0,              # -1 auto; 0 freeze; small positive to adapt
    shadow_value: int = 127,                     # MOG2 shadow value when detect_shadows=True
    # --- fusion
    use_union: bool = True,                      # union (OR) is recall-friendly; intersection is precision-friendly
    # --- postprocess
    open_ksize: int = 3,
    close_ksize: int = 7,
    # --- connected components filtering
    min_area: int = 80,
    max_area: Optional[int] = None,              # if None -> no upper bound
    min_wh: int = 5,
    max_aspect_ratio: float = 8.0,               # filter extremely thin noise
) -> CandidateGenResult:
    """
    Step3: motion-based candidate generation.

    Inputs:
      curr_gray: Step1 output for current frame (H,W) uint8
      prev_aligned: Step2 output aligned prev frame (H,W) uint8
      bg_subtractor: persistent MOG2 object (optional but recommended)
      valid_mask: optional mask to ignore subtitles/overlays (255 valid, 0 ignore)

    Outputs:
      mask: binary foreground mask (0/255)
      boxes: candidate bounding boxes (x,y,w,h)
      debug: stats to help tuning
    """
    curr_gray = _ensure_gray_u8(curr_gray)
    prev_aligned = _ensure_gray_u8(prev_aligned)

    H, W = curr_gray.shape[:2]
    debug: Dict[str, Any] = {}

    # apply valid mask (e.g., ignore subtitle region)
    curr_use = _apply_mask(curr_gray, valid_mask)
    prev_use = _apply_mask(prev_aligned, valid_mask)

    # ----------------------------
    # A) Aligned frame difference
    # ----------------------------
    diff = cv2.absdiff(curr_use, prev_use)

    # blur diff to suppress tiny flickering pixels on water/grass highlight
    if diff_blur_ksize and diff_blur_ksize > 1:
        if diff_blur_ksize % 2 == 0:
            diff_blur_ksize += 1
        diff = cv2.GaussianBlur(diff, (diff_blur_ksize, diff_blur_ksize), 0)

    _, diff_mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    debug["diff_thresh"] = diff_thresh
    debug["diff_fg_ratio"] = float(np.mean(diff_mask > 0))

    # --------------------------------
    # B) Background subtraction (MOG2)
    # --------------------------------
    bg_mask = None
    if bg_subtractor is not None:
        # MOG2 expects single channel or 3-channel; we pass gray
        raw_bg = bg_subtractor.apply(curr_use, learningRate=bg_learning_rate)

        # remove shadows: raw_bg can have 0(background), 127(shadow), 255(fg)
        if shadow_value is not None:
            bg_mask = np.where(raw_bg == shadow_value, 0, raw_bg).astype(np.uint8)
        else:
            bg_mask = raw_bg

        # binarize (some versions may output 0/255 already but keep safe)
        _, bg_mask = cv2.threshold(bg_mask, 200, 255, cv2.THRESH_BINARY)

        debug["bg_learning_rate"] = bg_learning_rate
        debug["bg_fg_ratio"] = float(np.mean(bg_mask > 0))
    else:
        debug["bg_warning"] = "bg_subtractor is None; using diff only"

    # ----------
    # Fusion
    # ----------
    if bg_mask is None:
        fused = diff_mask
        debug["fusion"] = "diff_only"
    else:
        if use_union:
            fused = cv2.bitwise_or(diff_mask, bg_mask)      # recall ↑ (more complete birds)
            debug["fusion"] = "OR(diff, bg)"
        else:
            fused = cv2.bitwise_and(diff_mask, bg_mask)     # precision ↑ (fewer false positives)
            debug["fusion"] = "AND(diff, bg)"

    # ----------------
    # Post-processing
    # ----------------
    fused = _binary_cleanup(fused, k_open=open_ksize, k_close=close_ksize)

    # optional: fill tiny gaps by dilation then erosion (close already does similar)
    # fused = cv2.dilate(fused, None, iterations=1)
    # fused = cv2.erode(fused, None, iterations=1)

    # -------------------------------
    # Connected components -> boxes
    # -------------------------------
    # Using connectedComponentsWithStats is often cleaner than contours for filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused, connectivity=8)

    boxes: List[Tuple[int, int, int, int]] = []
    kept = 0
    for label in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if w < min_wh or h < min_wh:
            continue
        ar = (max(w, h) / max(1, min(w, h)))
        if ar > max_aspect_ratio:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
        kept += 1

    debug["cc_total"] = int(num_labels - 1)
    debug["cc_kept"] = int(kept)
    debug["final_fg_ratio"] = float(np.mean(fused > 0))

    return CandidateGenResult(mask=fused, boxes=boxes, debug=debug)
