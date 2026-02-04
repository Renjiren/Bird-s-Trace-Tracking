# candidate_generation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

from preprocessing import ensure_gray_u8, apply_valid_mask_fill, odd, scaled_k


@dataclass(frozen=True)
class CandidateGenConfig:
    # ------------------------
    # Main trunk: diff_n + MAD
    # ------------------------
    eps: float = 3.0                 # normalized diff denominator
    diff_blur_ksize: int = 5         # blur on diff maps to suppress tiny sparkle
    mad_k: float = 8.0               # thr = median + k*(1.4826*MAD)
    thr_min: int = 8                 # clamp threshold to avoid too low
    thr_max: int = 80                # clamp threshold to avoid too high

    # specular suppression (soft mask)
    # weight applied to diff inside specular-like region (0..1)
    specular_weight: float = 0.35

    # only when specular is extreme, allow "conditional hard suppression"
    # (still not always; and only suppress spec region)
    spec_ratio_hard: float = 0.20
    fg_ratio_hard: float = 0.30

    # ------------------------
    # Optional branch A: diff_g (LoG-based) auto switch
    # ------------------------
    auto_enable_diff_g: bool = True
    edge_thr_u8: int = 30
    edge_density_thresh: float = 0.06
    spec_ratio_enable_diff_g: float = 0.05
    fg_ratio_enable_diff_g: float = 0.18
    diff_g_mad_k: float = 7.0
    fuse_and_when_complex: bool = True   # complex/flickery -> AND else OR

    # ------------------------
    # Optional branch B: KNN background subtractor auto switch
    # ------------------------
    auto_enable_bg: bool = True
    bg_history: int = 300
    bg_dist2_threshold: float = 400.0
    bg_detect_shadows: bool = True
    shadow_value: int = 127
    bg_thresh: int = 200

    moving_ema_alpha: float = 0.10
    moving_ema_enable_bg: float = 0.25
    fg_ratio_enable_bg: float = 0.10
    spec_ratio_enable_bg: float = 0.06

    bg_lr_still: float = 0.01
    bg_lr_when_fg_high: float = 0.0
    fg_ratio_freeze_bg: float = 0.18

    # ------------------------
    # NEW: BG-only component suppression (post-fusion)
    # ------------------------
    enable_bg_only_suppression: bool = True

    # We consider a component "bg-only" if diff support inside it is too small.
    bg_only_support_min: float = 0.06  # overlap(diff_main, component)/area(component)

    # If a component is bg-only, require its mean diff be not too close to threshold.
    # We drop bg-only comps whose mean_diff < (thr_n + margin).
    bg_only_mean_diff_margin: float = 2.0

    # BG-only comps must persist across frames to survive (strong filter for dynamic background).
    # With decay=0.85, 2 consecutive frames => ~1.85. So 1.6 roughly means ">=2 frames".
    bg_only_persistence_min: float = 1.60

    # Warmup before applying bg-only persistence gate
    bg_only_persistence_warmup: int = 5

    # Optional: dilate diff support a bit (tolerate small misalignment / morphology bridging)
    bg_only_support_dilate_ksize: int = 3

    # ------------------------
    # Morphology (adaptive scaling by resolution)
    # ------------------------
    kernel_scale_ref: int = 720
    open_ksize: int = 3
    close_ksize: int = 9
    bridge_dilate_ksize: int = 5
    bridge_close_ksize: int = 11
    use_bridge: bool = True
    dilate_ksize: int = 0

    # ------------------------
    # CC filtering
    # ------------------------
    min_area: int = 80
    max_area: Optional[int] = None
    min_wh: int = 5
    max_aspect_ratio: float = 10.0
    max_boxes: int = 50
    bbox_pad_frac: float = 0.20

    # nested box suppression
    enable_nested_suppression: bool = True
    nested_ioa_thresh: float = 0.92
    nested_area_ratio: float = 0.18

    # ------------------------
    # Lightweight temporal stability scoring
    # ------------------------
    use_persistence: bool = True
    persistence_decay: float = 0.85
    persistence_add: float = 1.0

    use_edge_change_score: bool = True

    # ------------------------
    # Optional global persistence filter (usually keep disabled; bg-only is the main win)
    # ------------------------
    enable_global_persistence_filter: bool = False
    global_persistence_min: float = 0.80
    global_persistence_warmup: int = 5


@dataclass
class CandidateGenResult:
    mask: np.ndarray
    boxes: List[Tuple[int, int, int, int]]
    debug: Dict[str, Any]


class MotionCandidateGenerator:
    """
    Per-video state holder:
    - KNN background model (optional/auto)
    - moving EMA to decide enable bg
    - persistence map for temporal stability scoring
    """

    def __init__(self, cfg: CandidateGenConfig):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorKNN(
            history=cfg.bg_history,
            dist2Threshold=cfg.bg_dist2_threshold,
            detectShadows=cfg.bg_detect_shadows,
        )
        self.moving_ema = 1.0
        self.persistence: Optional[np.ndarray] = None
        self.frame_idx = 0

    def reset(self):
        cfg = self.cfg
        self.bg = cv2.createBackgroundSubtractorKNN(
            history=cfg.bg_history,
            dist2Threshold=cfg.bg_dist2_threshold,
            detectShadows=cfg.bg_detect_shadows,
        )
        self.moving_ema = 1.0
        self.persistence = None
        self.frame_idx = 0

    def update_moving_ema(self, camera_moving: bool):
        a = float(np.clip(self.cfg.moving_ema_alpha, 0.01, 0.5))
        x = 1.0 if camera_moving else 0.0
        self.moving_ema = (1.0 - a) * float(self.moving_ema) + a * x


# -------------------- utils --------------------
def ratio_in_valid(mask_bool: np.ndarray, valid_mask: Optional[np.ndarray]) -> float:
    """
    All ratio stats are computed only inside valid region (exclude subtitles/borders).
    mask_bool: True/False
    valid_mask: 255 valid / 0 invalid
    """
    if valid_mask is None:
        return float(np.mean(mask_bool))
    v = (valid_mask > 0)
    vn = int(np.count_nonzero(v))
    if vn <= 0:
        return 0.0
    return float(np.mean(mask_bool[v]))


def mad_threshold_u8(
    img_u8: np.ndarray,
    mask_u8: Optional[np.ndarray],
    k: float,
    tmin: int,
    tmax: int
) -> Tuple[int, Dict[str, Any]]:
    """thr = median + k * (1.4826*MAD); clamp."""
    if mask_u8 is not None and np.any(mask_u8 > 0):
        vals = img_u8[mask_u8 > 0].astype(np.float32)
    else:
        vals = img_u8.reshape(-1).astype(np.float32)

    if vals.size == 0:
        thr = int(np.clip(25, tmin, tmax))
        return thr, {"median": None, "mad": None, "sigma": None, "thr": thr}

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)) + 1e-6)
    sigma = 1.4826 * mad
    thr = int(np.clip(med + float(k) * sigma, tmin, tmax))
    return thr, {"median": med, "mad": mad, "sigma": sigma, "thr": thr}


def binary_cleanup(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    out = mask
    if k_open and k_open > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k_open), odd(k_open)))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker, iterations=1)
    if k_close and k_close > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k_close), odd(k_close)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=1)
    return out


def bridge(mask: np.ndarray, k_dilate: int, k_close: int) -> np.ndarray:
    """Bridge broken parts: dilate -> close -> erode."""
    if k_dilate <= 1 and k_close <= 1:
        return mask
    kd = odd(max(1, k_dilate))
    kc = odd(max(1, k_close))
    ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    ker_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    x = cv2.dilate(mask, ker_d, iterations=1)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, ker_c, iterations=1)
    x = cv2.erode(x, ker_d, iterations=1)
    return x


def maybe_dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k and k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k), odd(k)))
        return cv2.dilate(mask, ker, iterations=1)
    return mask


def pad_box(x: int, y: int, w: int, h: int, H: int, W: int, pad_frac: float) -> Tuple[int, int, int, int]:
    px = int(round(w * pad_frac))
    py = int(round(h * pad_frac))
    x2 = max(0, x - px)
    y2 = max(0, y - py)
    x3 = min(W, x + w + px)
    y3 = min(H, y + h + py)
    return x2, y2, x3 - x2, y3 - y2


def ioa(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
    """Intersection over area of inner."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    ax1, ay1, ax2, ay2 = ix, iy, ix + iw, iy + ih
    bx1, by1, bx2, by2 = ox, oy, ox + ow, oy + oh
    cx1, cy1 = max(ax1, bx1), max(ay1, by1)
    cx2, cy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    area_in = max(1, iw * ih)
    return float(inter) / float(area_in)


def suppress_nested_boxes(
    boxes: List[Tuple[int, int, int, int]],
    ioa_thresh: float,
    area_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """Remove small boxes that are almost completely inside a bigger box."""
    if len(boxes) <= 1:
        return boxes

    areas = [b[2] * b[3] for b in boxes]
    order = np.argsort(-np.array(areas))  # big -> small
    keep = [True] * len(boxes)

    for i in range(len(order)):
        bi = boxes[int(order[i])]
        ai = areas[int(order[i])]
        if not keep[int(order[i])]:
            continue
        for j in range(i + 1, len(order)):
            idxj = int(order[j])
            if not keep[idxj]:
                continue
            bj = boxes[idxj]
            aj = areas[idxj]
            if aj <= area_ratio * ai:
                if _ioa(bj, bi) >= ioa_thresh:
                    keep[idxj] = False

    return [b for b, k in zip(boxes, keep) if k]


def edge_density_from_log(log_u8: np.ndarray, thr: int, valid_mask: Optional[np.ndarray]) -> float:
    e = (log_u8 >= int(thr))
    return ratio_in_valid(e, valid_mask)


def normalized_diff_u8(curr_u8: np.ndarray, prev_u8: np.ndarray, eps: float) -> np.ndarray:
    """diff_n = |a-b| / (a+b+eps) mapped to 0..255."""
    a = curr_u8.astype(np.float32)
    b = prev_u8.astype(np.float32)
    num = np.abs(a - b)
    den = (a + b + float(eps))
    dn = num / den
    out = np.clip(dn * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def apply_spec_weight(
    diff_u8: np.ndarray,
    spec_mask: Optional[np.ndarray],
    weight: float,
    valid_mask: Optional[np.ndarray]
) -> np.ndarray:
    """
    spec_mask: 255 normal, 0 specular-like
    Apply down-weight inside specular region (soft, not hard ignore).
    Only applies inside valid region (subtitle invalid area excluded).
    """
    if spec_mask is None:
        return diff_u8
    w = float(np.clip(weight, 0.0, 1.0))
    if w >= 0.999:
        return diff_u8

    out = diff_u8.astype(np.float32)
    spec = (spec_mask == 0)
    if valid_mask is not None:
        spec = spec & (valid_mask > 0)

    out[spec] *= w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def update_persistence(gen: MotionCandidateGenerator, fg_mask: np.ndarray, valid_mask: Optional[np.ndarray]):
    cfg = gen.cfg
    if not cfg.use_persistence:
        return
    m = (fg_mask > 0).astype(np.float32)
    if valid_mask is not None:
        m = m * (valid_mask > 0).astype(np.float32)

    if gen.persistence is None or gen.persistence.shape != m.shape:
        gen.persistence = m * float(cfg.persistence_add)
        return
    gen.persistence = float(cfg.persistence_decay) * gen.persistence + float(cfg.persistence_add) * m


def box_score_persistence(gen: MotionCandidateGenerator, box: Tuple[int, int, int, int]) -> float:
    if gen.persistence is None:
        return 0.0
    x, y, w, h = box
    roi = gen.persistence[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi))


def box_score_edge_change(
    curr_log: Optional[np.ndarray],
    prev_log_aligned: Optional[np.ndarray],
    box: Tuple[int, int, int, int]
) -> float:
    if curr_log is None or prev_log_aligned is None:
        return 0.0
    x, y, w, h = box
    a = curr_log[y:y + h, x:x + w]
    b = prev_log_aligned[y:y + h, x:x + w]
    if a.size == 0:
        return 0.0
    d = cv2.absdiff(a, b)
    return float(np.mean(d))


def component_mean_from_labels(img_u8: np.ndarray, labels: np.ndarray, lab: int) -> float:
    m = (labels == lab)
    if not np.any(m):
        return 0.0
    vals = img_u8[m]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def component_overlap_ratio(mask_u8: np.ndarray, labels: np.ndarray, lab: int) -> float:
    """
    overlap ratio between (mask_u8>0) and component pixels (labels==lab).
    i.e., count(mask & comp)/count(comp)
    """
    comp = (labels == lab)
    denom = int(np.count_nonzero(comp))
    if denom <= 0:
        return 0.0
    num = int(np.count_nonzero((mask_u8 > 0) & comp))
    return float(num) / float(denom)


# -------------------- main --------------------
def generate_motion_candidates(
    curr_intensity: np.ndarray,
    prev_intensity_aligned: np.ndarray,
    valid_mask: Optional[np.ndarray],
    spec_mask: Optional[np.ndarray],
    gen: MotionCandidateGenerator,
    camera_moving: bool,
    curr_log: Optional[np.ndarray] = None,
    prev_log_aligned: Optional[np.ndarray] = None,
) -> CandidateGenResult:
    """
    Step3:
    - Main trunk: diff_n + MAD threshold + spec down-weight + morphology + CC + nested suppression
    - Auto optional: diff_g (LoG diff) & KNN bg subtractor
    - bg-only component suppression + persistence filtering to reduce proposals in dynamic scenes

    Inputs:
      curr_intensity: uint8 intensity at time t
      prev_intensity_aligned: uint8 intensity warped to time t (from Step2)
      valid_mask: 255 valid / 0 invalid (subtitle/border hard mask)
      spec_mask: 255 normal / 0 specular-like (soft mask)
      camera_moving: Step2 output
      curr_log / prev_log_aligned: uint8 LoG features (optional but recommended)
    """
    cfg = gen.cfg
    gen.frame_idx += 1
    gen.update_moving_ema(camera_moving)

    curr = ensure_gray_u8(curr_intensity)
    prev = ensure_gray_u8(prev_intensity_aligned)
    H, W = curr.shape[:2]

    debug: Dict[str, Any] = {
        "frame_idx": int(gen.frame_idx),
        "camera_moving": bool(camera_moving),
        "moving_ema": float(gen.moving_ema),
    }

    # ---------- hard valid handling ----------
    # Use fill instead of hard 0 holes to avoid strong edges near subtitles/borders.
    curr_use = apply_valid_mask_fill(curr, valid_mask)
    prev_use = apply_valid_mask_fill(prev, valid_mask)

    # ------------------------
    # Main trunk: diff_n + MAD
    # ------------------------
    diff_n = normalized_diff_u8(curr_use, prev_use, cfg.eps)

    if cfg.diff_blur_ksize and cfg.diff_blur_ksize > 1:
        k = odd(cfg.diff_blur_ksize)
        diff_n = cv2.GaussianBlur(diff_n, (k, k), 0)

    # spec down-weight (soft)
    diff_n_w = apply_spec_weight(diff_n, spec_mask, cfg.specular_weight, valid_mask)

    thr_n, thr_dbg = mad_threshold_u8(diff_n_w, valid_mask, cfg.mad_k, cfg.thr_min, cfg.thr_max)
    diff_mask = (diff_n_w >= thr_n).astype(np.uint8) * 255

    fg_ratio_n = ratio_in_valid(diff_mask > 0, valid_mask)
    spec_ratio = ratio_in_valid(
        (spec_mask == 0) if spec_mask is not None else np.zeros((H, W), bool),
        valid_mask
    )

    debug["diff_n"] = {
        "thr": int(thr_n),
        "stats": thr_dbg,
        "fg_ratio_valid": float(fg_ratio_n),
        "specular_weight": float(cfg.specular_weight),
    }
    debug["spec_ratio_valid"] = float(spec_ratio)

    # ------------------------
    # Auto decide: enable diff_g?
    # ------------------------
    enable_diff_g = False
    scene_complex = False
    edge_density = None

    if cfg.auto_enable_diff_g and (curr_log is not None) and (prev_log_aligned is not None):
        clog = ensure_gray_u8(curr_log)
        edge_density = edge_density_from_log(clog, cfg.edge_thr_u8, valid_mask)
        scene_complex = (edge_density >= cfg.edge_density_thresh)

        enable_diff_g = bool(
            scene_complex
            or (spec_ratio >= cfg.spec_ratio_enable_diff_g)
            or (fg_ratio_n >= cfg.fg_ratio_enable_diff_g)
        )

    debug["auto_diff_g"] = {
        "edge_density_valid": None if edge_density is None else float(edge_density),
        "scene_complex": bool(scene_complex),
        "enable_diff_g": bool(enable_diff_g),
    }

    # ------------------------
    # Optional branch A: diff_g
    # ------------------------
    if enable_diff_g:
        clog = ensure_gray_u8(curr_log)
        plog = ensure_gray_u8(prev_log_aligned)
        diff_g = cv2.absdiff(clog, plog)

        if cfg.diff_blur_ksize and cfg.diff_blur_ksize > 1:
            k = odd(cfg.diff_blur_ksize)
            diff_g = cv2.GaussianBlur(diff_g, (k, k), 0)

        diff_g_w = apply_spec_weight(diff_g, spec_mask, cfg.specular_weight, valid_mask)

        thr_g, thr_g_dbg = mad_threshold_u8(diff_g_w, valid_mask, cfg.diff_g_mad_k, cfg.thr_min, cfg.thr_max)
        diff_g_mask = (diff_g_w >= thr_g).astype(np.uint8) * 255
        fg_ratio_g = ratio_in_valid(diff_g_mask > 0, valid_mask)

        debug["diff_g"] = {
            "thr": int(thr_g),
            "stats": thr_g_dbg,
            "fg_ratio_valid": float(fg_ratio_g),
        }

        use_and = bool(cfg.fuse_and_when_complex and (scene_complex or spec_ratio >= cfg.spec_ratio_enable_diff_g))
        if use_and:
            diff_main = cv2.bitwise_and(diff_mask, diff_g_mask)
            debug["diff_fusion"] = "AND"
        else:
            diff_main = cv2.bitwise_or(diff_mask, diff_g_mask)
            debug["diff_fusion"] = "OR"
    else:
        diff_main = diff_mask
        debug["diff_fusion"] = "diff_n_only"

    fg_ratio_main = ratio_in_valid(diff_main > 0, valid_mask)
    debug["diff_main_fg_ratio_valid"] = float(fg_ratio_main)

    # ------------------------
    # Auto decide: enable bg subtractor?
    # ------------------------
    enable_bg = False
    if cfg.auto_enable_bg:
        mostly_still = (gen.moving_ema <= cfg.moving_ema_enable_bg)
        enable_bg = bool(
            mostly_still and (
                fg_ratio_main >= cfg.fg_ratio_enable_bg
                or spec_ratio >= cfg.spec_ratio_enable_bg
            )
        )
        debug["auto_bg"] = {"mostly_still": bool(mostly_still), "enable_bg": bool(enable_bg)}
    else:
        debug["auto_bg"] = {"mostly_still": None, "enable_bg": False}

    # ------------------------
    # Optional branch B: KNN background subtractor
    # ------------------------
    bg_mask = None
    if enable_bg:
        if camera_moving:
            lr = 0.0
        else:
            lr = float(cfg.bg_lr_still)
            if fg_ratio_main >= cfg.fg_ratio_freeze_bg:
                lr = float(cfg.bg_lr_when_fg_high)

        raw = gen.bg.apply(curr_use, learningRate=lr)

        if cfg.shadow_value is not None:
            raw = np.where(raw == cfg.shadow_value, 0, raw).astype(np.uint8)

        _, bg_mask = cv2.threshold(raw, int(cfg.bg_thresh), 255, cv2.THRESH_BINARY)

        debug["bg"] = {
            "learningRate": float(lr),
            "fg_ratio_valid": float(ratio_in_valid(bg_mask > 0, valid_mask)),
            "bg_thresh": int(cfg.bg_thresh),
        }

    # ------------------------
    # Fusion (high recall) + spec guard (conditional hard)
    # ------------------------
    if bg_mask is None:
        fused = diff_main
        debug["fusion"] = "diff_only"
    else:
        fused = cv2.bitwise_or(diff_main, bg_mask)
        debug["fusion"] = "diff_or_bg"

    fused_fg_ratio = ratio_in_valid(fused > 0, valid_mask)
    debug["fused_fg_ratio_valid_before_spec_guard"] = float(fused_fg_ratio)

    # Conditional hard suppression: only in extreme glare-dominant cases; suppress only spec region.
    if spec_mask is not None:
        if (spec_ratio >= cfg.spec_ratio_hard) or (fused_fg_ratio >= cfg.fg_ratio_hard):
            fused = fused.copy()
            cond = (spec_mask == 0)
            if valid_mask is not None:
                cond = cond & (valid_mask > 0)
            fused[cond] = 0
            debug["spec_guard"] = "hard_suppress_spec_in_extreme"
        else:
            debug["spec_guard"] = "soft_only"
    else:
        debug["spec_guard"] = "no_spec_mask"

    # Output must strictly exclude invalid region
    if valid_mask is not None:
        fused = cv2.bitwise_and(fused, valid_mask)
        diff_main = cv2.bitwise_and(diff_main, valid_mask)
        if bg_mask is not None:
            bg_mask = cv2.bitwise_and(bg_mask, valid_mask)

    # ------------------------
    # Morphology: adaptive kernel sizes by resolution
    # ------------------------
    k_open = scaled_k(cfg.open_ksize, H, W, cfg.kernel_scale_ref)
    k_close = scaled_k(cfg.close_ksize, H, W, cfg.kernel_scale_ref)

    fused_m = binary_cleanup(fused, k_open, k_close)

    if cfg.use_bridge:
        k_bd = scaled_k(cfg.bridge_dilate_ksize, H, W, cfg.kernel_scale_ref)
        k_bc = scaled_k(cfg.bridge_close_ksize, H, W, cfg.kernel_scale_ref)
        fused_m = bridge(fused_m, k_bd, k_bc)

    if cfg.dilate_ksize:
        fused_m = maybe_dilate(fused_m, scaled_k(cfg.dilate_ksize, H, W, cfg.kernel_scale_ref))

    # Ensure hard constraint again
    if valid_mask is not None:
        fused_m = cv2.bitwise_and(fused_m, valid_mask)

    debug["morph"] = {
        "k_open": int(k_open),
        "k_close": int(k_close),
        "bridge": bool(cfg.use_bridge),
    }

    # Update persistence after morphology (more stable)
    update_persistence(gen, fused_m, valid_mask)

    # ------------------------
    # NEW: Build diff support mask for bg-only suppression
    # ------------------------
    diff_support = diff_main.copy()
    # Optional dilation: tolerate small misalignment and morphology bridging
    if cfg.bg_only_support_dilate_ksize and cfg.bg_only_support_dilate_ksize > 1:
        kd = scaled_k(cfg.bg_only_support_dilate_ksize, H, W, cfg.kernel_scale_ref)
        diff_support = maybe_dilate(diff_support, kd)
    if valid_mask is not None:
        diff_support = cv2.bitwise_and(diff_support, valid_mask)

    # ------------------------
    # Connected components -> boxes (with BG-only suppression + persistence filter)
    # ------------------------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fused_m, connectivity=8)

    suppressed_bg_only = 0
    kept_components = 0
    comps_for_box: List[Tuple[int, int, int, int, int, float, float, float]] = []
    # (area, x, y, w, h, diff_support_ratio, mean_diff, persistence_mean)

    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        x, y, w, h, area = int(x), int(y), int(w), int(h), int(area)

        if area < cfg.min_area:
            continue
        if cfg.max_area is not None and area > cfg.max_area:
            continue
        if w < cfg.min_wh or h < cfg.min_wh:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > cfg.max_aspect_ratio:
            continue

        # --- BG-only suppression (only meaningful when bg_mask participated) ---
        diff_support_ratio = component_overlap_ratio(diff_support, labels, lab)

        # Mean diff response (use weighted diff_n_w, robust against glare)
        mean_diff = component_mean_from_labels(diff_n_w, labels, lab)

        # Persistence mean (component-level)
        if gen.persistence is None:
            pers_mean = 0.0
        else:
            pm = gen.persistence[labels == lab]
            pers_mean = float(np.mean(pm)) if pm.size > 0 else 0.0

        is_bg_only = (bg_mask is not None) and (diff_support_ratio < float(cfg.bg_only_support_min))

        if (bg_mask is not None) and cfg.enable_bg_only_suppression and is_bg_only:
            # Apply persistence gate after warmup to kill dynamic background fragments
            warm = int(cfg.bg_only_persistence_warmup)
            pers_gate_on = (gen.frame_idx >= warm)

            # Low-diff gate: bg-only components with mean diff near threshold are likely leaves/waves.
            low_diff = (mean_diff < float(thr_n) + float(cfg.bg_only_mean_diff_margin))

            # Persistence gate: require >=2 frames (default)
            low_pers = pers_gate_on and (pers_mean < float(cfg.bg_only_persistence_min))

            # Suppress if bg-only AND (low diff) AND (low persistence after warmup)
            if low_diff and low_pers:
                suppressed_bg_only += 1
                continue

        # Optional global persistence filter (usually OFF; use cautiously)
        if cfg.enable_global_persistence_filter and gen.persistence is not None:
            if gen.frame_idx >= int(cfg.global_persistence_warmup):
                if pers_mean < float(cfg.global_persistence_min):
                    continue

        comps_for_box.append((area, x, y, w, h, diff_support_ratio, mean_diff, pers_mean))
        kept_components += 1

    comps_for_box.sort(key=lambda t: t[0], reverse=True)

    boxes: List[Tuple[int, int, int, int]] = []
    comp_dbg_top: List[Dict[str, Any]] = []

    for idx, (area, x, y, w, h, dsr, md, pm) in enumerate(comps_for_box):
        x2, y2, w2, h2 = pad_box(x, y, w, h, H, W, cfg.bbox_pad_frac)
        boxes.append((x2, y2, w2, h2))

        if idx < 12:
            comp_dbg_top.append({
                "idx": int(idx),
                "area": int(area),
                "bbox": [int(x), int(y), int(w), int(h)],
                "diff_support_ratio": float(dsr),
                "mean_diff": float(md),
                "persistence_mean": float(pm),
            })

        if len(boxes) >= cfg.max_boxes * 3:
            break

    debug["cc"] = {
        "total": int(num - 1),
        "kept_after_geom": int(kept_components),
        "suppressed_bg_only": int(suppressed_bg_only),
        "final_fg_ratio_valid": float(ratio_in_valid(fused_m > 0, valid_mask)),
        "top_components": comp_dbg_top,
    }

    # ------------------------
    # Parent-child suppression
    # ------------------------
    if cfg.enable_nested_suppression and len(boxes) > 1:
        before = len(boxes)
        boxes = suppress_nested_boxes(boxes, cfg.nested_ioa_thresh, cfg.nested_area_ratio)
        debug["nested_suppression"] = {"before": int(before), "after": int(len(boxes))}
    else:
        debug["nested_suppression"] = {"before": int(len(boxes)), "after": int(len(boxes))}

    # ------------------------
    # Lightweight scoring + truncation
    # ------------------------
    scored = []
    for b in boxes:
        s_p = box_score_persistence(gen, b) if cfg.use_persistence else 0.0
        s_e = box_score_edge_change(curr_log, prev_log_aligned, b) if cfg.use_edge_change_score else 0.0
        area = float(b[2] * b[3])
        scored.append((b, s_p, s_e, area))

    scored.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
    boxes_sorted = [t[0] for t in scored[:cfg.max_boxes]]

    debug["scoring"] = {
        "use_persistence": bool(cfg.use_persistence),
        "use_edge_change_score": bool(cfg.use_edge_change_score),
        "kept": int(len(boxes_sorted)),
        "top_scores": [
            {"idx": i, "persistence": float(t[1]), "edge_change": float(t[2]), "area": float(t[3])}
            for i, t in enumerate(scored[:min(10, len(scored))])
        ],
    }

    return CandidateGenResult(mask=fused_m, boxes=boxes_sorted, debug=debug)


def draw_overlay(gray_u8: np.ndarray, boxes: List[Tuple[int, int, int, int]], max_draw: int = 50) -> np.ndarray:
    """Visualization helper (grayscale -> BGR with boxes)."""
    g = ensure_gray_u8(gray_u8)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes[:max_draw]):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis