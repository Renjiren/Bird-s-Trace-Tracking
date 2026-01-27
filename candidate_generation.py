# candidate_generation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from preprocessing import ensure_gray_u8, apply_valid_mask_fill


@dataclass(frozen=True)
class CandidateGenConfig:
    # ------------------------
    # Main trunk: diff_n + MAD
    # ------------------------
    eps: float = 3.0  # for normalized diff denominator
    diff_blur_ksize: int = 5  # blur on diff maps to suppress tiny sparkle
    mad_k: float = 8.0  # thr = median + k * (1.4826*MAD)
    thr_min: int = 8   # clamp threshold to avoid too low
    thr_max: int = 80  # clamp threshold to avoid too high

    # specular suppression (NOT hard ignore)
    # weight applied to diff inside specular-like region (0..1)
    specular_weight: float = 0.35
    # only when specular is extreme, allow "conditional hard suppression" (still not always)
    spec_ratio_hard: float = 0.20
    fg_ratio_hard: float = 0.30

    # ------------------------
    # Optional branch A: diff_g (LoG-based) auto switch
    # ------------------------
    auto_enable_diff_g: bool = True
    edge_thr_u8: int = 30            # for edge density on LoG
    edge_density_thresh: float = 0.06  # above -> complex texture
    spec_ratio_enable_diff_g: float = 0.05
    fg_ratio_enable_diff_g: float = 0.18
    diff_g_mad_k: float = 7.0        # MAD threshold for diff_g
    # fusion mode when diff_g enabled:
    # - if scene is flickery/complex -> AND for precision
    # - else -> OR for recall
    fuse_and_when_complex: bool = True

    # ------------------------
    # Optional branch B: KNN background subtractor auto switch
    # ------------------------
    auto_enable_bg: bool = True
    bg_history: int = 300
    bg_dist2_threshold: float = 400.0
    bg_detect_shadows: bool = True
    shadow_value: int = 127
    bg_thresh: int = 200

    # if camera mostly still AND dynamic texture persists -> enable bg
    moving_ema_alpha: float = 0.10
    moving_ema_enable_bg: float = 0.25     # smaller -> more still
    fg_ratio_enable_bg: float = 0.10
    spec_ratio_enable_bg: float = 0.06

    # learningRate control (avoid contaminating background)
    bg_lr_still: float = 0.01
    bg_lr_when_fg_high: float = 0.0
    fg_ratio_freeze_bg: float = 0.18

    # ------------------------
    # Morphology (adaptive scaling by resolution)
    # ------------------------
    kernel_scale_ref: int = 720  # reference min(H,W) for kernel sizes
    open_ksize: int = 3
    close_ksize: int = 9
    bridge_dilate_ksize: int = 5   # bridge: dilate->close->erode
    bridge_close_ksize: int = 11
    use_bridge: bool = True
    dilate_ksize: int = 0          # optional expand

    # ------------------------
    # CC filtering
    # ------------------------
    min_area: int = 80
    max_area: Optional[int] = None
    min_wh: int = 5
    max_aspect_ratio: float = 10.0
    max_boxes: int = 50
    bbox_pad_frac: float = 0.20

    # nested box suppression (parent-child)
    enable_nested_suppression: bool = True
    nested_ioa_thresh: float = 0.92   # small box mostly inside big box
    nested_area_ratio: float = 0.18   # small area <= ratio * big area => drop

    # ------------------------
    # Lightweight temporal stability scoring (no optical flow)
    # ------------------------
    use_persistence: bool = True
    persistence_decay: float = 0.85
    persistence_add: float = 1.0

    # optional edge-change score (uses LoG diff if provided)
    use_edge_change_score: bool = True


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
        self.moving_ema = 1.0  # start "moving" pessimistically
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


def _odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k + (k % 2 == 0)


def _scaled_k(base: int, H: int, W: int, ref: int) -> int:
    s = float(min(H, W)) / float(max(1, ref))
    return _odd(max(1, int(round(base * s))))


def _mad_threshold_u8(img_u8: np.ndarray, mask_u8: Optional[np.ndarray], k: float, tmin: int, tmax: int) -> Tuple[int, Dict[str, Any]]:
    """thr = median + k * (1.4826*MAD); clamp."""
    if mask_u8 is not None and np.any(mask_u8 > 0):
        vals = img_u8[mask_u8 > 0].astype(np.float32)
    else:
        vals = img_u8.reshape(-1).astype(np.float32)

    if vals.size == 0:
        thr = int(np.clip(25, tmin, tmax))
        return thr, {"median": None, "mad": None, "thr": thr}

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)) + 1e-6)
    sigma = 1.4826 * mad
    thr = int(np.clip(med + float(k) * sigma, tmin, tmax))
    return thr, {"median": med, "mad": mad, "sigma": sigma, "thr": thr}


def _binary_cleanup(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    out = mask
    if k_open and k_open > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k_open), _odd(k_open)))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker, iterations=1)
    if k_close and k_close > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k_close), _odd(k_close)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=1)
    return out


def _bridge(mask: np.ndarray, k_dilate: int, k_close: int) -> np.ndarray:
    """Bridge broken parts: dilate -> close -> erode."""
    if k_dilate <= 1 and k_close <= 1:
        return mask
    kd = _odd(max(1, k_dilate))
    kc = _odd(max(1, k_close))
    ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    ker_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    x = cv2.dilate(mask, ker_d, iterations=1)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, ker_c, iterations=1)
    x = cv2.erode(x, ker_d, iterations=1)
    return x


def _maybe_dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k and k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
        return cv2.dilate(mask, ker, iterations=1)
    return mask


def _pad_box(x: int, y: int, w: int, h: int, H: int, W: int, pad_frac: float) -> Tuple[int, int, int, int]:
    px = int(round(w * pad_frac))
    py = int(round(h * pad_frac))
    x2 = max(0, x - px)
    y2 = max(0, y - py)
    x3 = min(W, x + w + px)
    y3 = min(H, y + h + py)
    return x2, y2, x3 - x2, y3 - y2


def _ioa(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
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


def _suppress_nested_boxes(
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


def _edge_density_from_log(log_u8: np.ndarray, thr: int) -> float:
    return float(np.mean(log_u8 >= int(thr)))


def _normalized_diff_u8(curr_u8: np.ndarray, prev_u8: np.ndarray, eps: float) -> np.ndarray:
    """diff_n = |a-b| / (a+b+eps) mapped to 0..255."""
    a = curr_u8.astype(np.float32)
    b = prev_u8.astype(np.float32)
    num = np.abs(a - b)
    den = (a + b + float(eps))
    dn = num / den
    out = np.clip(dn * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def _apply_spec_weight(diff_u8: np.ndarray, spec_mask: Optional[np.ndarray], weight: float) -> np.ndarray:
    """
    spec_mask: 255 normal, 0 specular-like
    Apply down-weight inside specular region, NOT hard ignore.
    """
    if spec_mask is None:
        return diff_u8
    w = float(np.clip(weight, 0.0, 1.0))
    if w >= 0.999:
        return diff_u8
    out = diff_u8.astype(np.float32)
    spec = (spec_mask == 0)
    out[spec] *= w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _update_persistence(gen: MotionCandidateGenerator, fg_mask: np.ndarray):
    cfg = gen.cfg
    if not cfg.use_persistence:
        return
    m = (fg_mask > 0).astype(np.float32)
    if gen.persistence is None or gen.persistence.shape != m.shape:
        gen.persistence = m * float(cfg.persistence_add)
        return
    gen.persistence = float(cfg.persistence_decay) * gen.persistence + float(cfg.persistence_add) * m


def _box_score_persistence(gen: MotionCandidateGenerator, box: Tuple[int, int, int, int]) -> float:
    if gen.persistence is None:
        return 0.0
    x, y, w, h = box
    roi = gen.persistence[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi))


def _box_score_edge_change(curr_log: Optional[np.ndarray], prev_log_aligned: Optional[np.ndarray], box: Tuple[int, int, int, int]) -> float:
    if curr_log is None or prev_log_aligned is None:
        return 0.0
    x, y, w, h = box
    a = curr_log[y:y+h, x:x+w]
    b = prev_log_aligned[y:y+h, x:x+w]
    if a.size == 0:
        return 0.0
    d = cv2.absdiff(a, b)
    return float(np.mean(d))


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
    - Lightweight temporal stability scoring (persistence), no optical flow

    Inputs:
      curr_intensity: uint8 intensity (preprocessed) at time t
      prev_intensity_aligned: uint8 intensity warped to time t coordinates (from Step2)
      valid_mask: 255 valid / 0 ignore
      spec_mask: 255 normal / 0 specular-like (from Step1)
      camera_moving: Step2 output
      curr_log / prev_log_aligned: uint8 LoG features (optional but recommended for diff_g and edge-change score)
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

    # Use valid_mask for fill to avoid hard edges from subtitles/borders
    curr_use = apply_valid_mask_fill(curr, valid_mask)
    prev_use = apply_valid_mask_fill(prev, valid_mask)

    # ------------------------
    # Main trunk: diff_n + MAD
    # ------------------------
    diff_n = _normalized_diff_u8(curr_use, prev_use, cfg.eps)

    # optional blur to suppress sparkle points
    if cfg.diff_blur_ksize and cfg.diff_blur_ksize > 1:
        k = _odd(cfg.diff_blur_ksize)
        diff_n = cv2.GaussianBlur(diff_n, (k, k), 0)

    # spec down-weight (NOT hard ignore)
    diff_n_w = _apply_spec_weight(diff_n, spec_mask, cfg.specular_weight)

    thr_n, thr_dbg = _mad_threshold_u8(diff_n_w, valid_mask, cfg.mad_k, cfg.thr_min, cfg.thr_max)
    diff_mask = (diff_n_w >= thr_n).astype(np.uint8) * 255

    fg_ratio_n = float(np.mean(diff_mask > 0))
    spec_ratio = float(np.mean(spec_mask == 0)) if spec_mask is not None else 0.0

    debug["diff_n"] = {
        "thr": int(thr_n),
        "stats": thr_dbg,
        "fg_ratio": fg_ratio_n,
        "specular_weight": float(cfg.specular_weight),
    }
    debug["spec_ratio"] = spec_ratio

    # ------------------------
    # Auto decide: enable diff_g?
    # ------------------------
    edge_density = None
    scene_complex = False
    enable_diff_g = False

    if cfg.auto_enable_diff_g and (curr_log is not None) and (prev_log_aligned is not None):
        clog = ensure_gray_u8(curr_log)
        plog = ensure_gray_u8(prev_log_aligned)
        edge_density = _edge_density_from_log(clog, cfg.edge_thr_u8)
        scene_complex = (edge_density >= cfg.edge_density_thresh)

        enable_diff_g = (
            scene_complex or
            (spec_ratio >= cfg.spec_ratio_enable_diff_g) or
            (fg_ratio_n >= cfg.fg_ratio_enable_diff_g)
        )

        debug["auto_diff_g"] = {
            "edge_density": float(edge_density),
            "scene_complex": bool(scene_complex),
            "enable_diff_g": bool(enable_diff_g),
        }
    else:
        debug["auto_diff_g"] = {
            "edge_density": None,
            "scene_complex": None,
            "enable_diff_g": False,
        }

    # ------------------------
    # Optional branch A: diff_g (LoG-based)
    # ------------------------
    if enable_diff_g:
        clog = ensure_gray_u8(curr_log)
        plog = ensure_gray_u8(prev_log_aligned)
        diff_g = cv2.absdiff(clog, plog)

        if cfg.diff_blur_ksize and cfg.diff_blur_ksize > 1:
            k = _odd(cfg.diff_blur_ksize)
            diff_g = cv2.GaussianBlur(diff_g, (k, k), 0)

        # spec down-weight here too
        diff_g_w = _apply_spec_weight(diff_g, spec_mask, cfg.specular_weight)

        thr_g, thr_g_dbg = _mad_threshold_u8(diff_g_w, valid_mask, cfg.diff_g_mad_k, cfg.thr_min, cfg.thr_max)
        diff_g_mask = (diff_g_w >= thr_g).astype(np.uint8) * 255
        fg_ratio_g = float(np.mean(diff_g_mask > 0))

        debug["diff_g"] = {
            "thr": int(thr_g),
            "stats": thr_g_dbg,
            "fg_ratio": fg_ratio_g,
        }

        # Fuse with trunk diff_n
        # Complex/flickery -> AND (precision), else OR (recall)
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

    fg_ratio_main = float(np.mean(diff_main > 0))
    debug["diff_main_fg_ratio"] = fg_ratio_main

    # ------------------------
    # Auto decide: enable bg subtractor?
    # ------------------------
    enable_bg = False
    if cfg.auto_enable_bg:
        mostly_still = (gen.moving_ema <= cfg.moving_ema_enable_bg)
        enable_bg = bool(
            mostly_still and (
                fg_ratio_main >= cfg.fg_ratio_enable_bg or
                spec_ratio >= cfg.spec_ratio_enable_bg
            )
        )
        debug["auto_bg"] = {
            "mostly_still": bool(mostly_still),
            "enable_bg": bool(enable_bg),
        }
    else:
        debug["auto_bg"] = {"mostly_still": None, "enable_bg": False}

    # ------------------------
    # Optional branch B: KNN background subtractor
    # ------------------------
    bg_mask = None
    if enable_bg:
        # learningRate: freeze when camera moving OR foreground too high
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
            "fg_ratio": float(np.mean(bg_mask > 0)),
            "bg_thresh": int(cfg.bg_thresh),
        }

    # ------------------------
    # Fusion (high recall then suppress specular extreme)
    # ------------------------
    if bg_mask is None:
        fused = diff_main
        debug["fusion"] = "diff_only"
    else:
        fused = cv2.bitwise_or(diff_main, bg_mask)
        debug["fusion"] = "diff_or_bg"

    fused_fg_ratio = float(np.mean(fused > 0))
    debug["fused_fg_ratio_before_spec_guard"] = fused_fg_ratio

    # Conditional hard suppression ONLY when specular dominates (extreme cases)
    if spec_mask is not None:
        if (spec_ratio >= cfg.spec_ratio_hard) or (fused_fg_ratio >= cfg.fg_ratio_hard):
            fused = fused.copy()
            fused[spec_mask == 0] = 0
            debug["spec_guard"] = "hard_suppress_in_extreme"
        else:
            debug["spec_guard"] = "soft_only"
    else:
        debug["spec_guard"] = "no_spec_mask"

    # ------------------------
    # Morphology: adaptive kernel sizes by resolution
    # ------------------------
    k_open = _scaled_k(cfg.open_ksize, H, W, cfg.kernel_scale_ref)
    k_close = _scaled_k(cfg.close_ksize, H, W, cfg.kernel_scale_ref)

    fused = _binary_cleanup(fused, k_open, k_close)

    if cfg.use_bridge:
        k_bd = _scaled_k(cfg.bridge_dilate_ksize, H, W, cfg.kernel_scale_ref)
        k_bc = _scaled_k(cfg.bridge_close_ksize, H, W, cfg.kernel_scale_ref)
        fused = _bridge(fused, k_bd, k_bc)

    fused = _maybe_dilate(fused, _scaled_k(cfg.dilate_ksize, H, W, cfg.kernel_scale_ref) if cfg.dilate_ksize else 0)

    debug["morph"] = {
        "k_open": int(k_open),
        "k_close": int(k_close),
        "bridge": bool(cfg.use_bridge),
    }

    # Update persistence after morphology (more stable)
    _update_persistence(gen, fused)

    # ------------------------
    # Connected components -> boxes
    # ------------------------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
    comps = []
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        comps.append((int(area), int(x), int(y), int(w), int(h)))
    comps.sort(key=lambda t: t[0], reverse=True)

    boxes: List[Tuple[int, int, int, int]] = []
    for area, x, y, w, h in comps:
        if area < cfg.min_area:
            continue
        if cfg.max_area is not None and area > cfg.max_area:
            continue
        if w < cfg.min_wh or h < cfg.min_wh:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > cfg.max_aspect_ratio:
            continue

        x2, y2, w2, h2 = _pad_box(x, y, w, h, H, W, cfg.bbox_pad_frac)
        boxes.append((x2, y2, w2, h2))
        if len(boxes) >= cfg.max_boxes * 3:
            # collect more, we'll sort by stability score and then cut
            break

    debug["cc"] = {
        "total": int(num - 1),
        "raw_kept": int(len(boxes)),
        "final_fg_ratio": float(np.mean(fused > 0)),
    }

    # ------------------------
    # Parent-child suppression (remove nested tiny boxes)
    # ------------------------
    if cfg.enable_nested_suppression and len(boxes) > 1:
        before = len(boxes)
        boxes = _suppress_nested_boxes(boxes, cfg.nested_ioa_thresh, cfg.nested_area_ratio)
        debug["nested_suppression"] = {"before": int(before), "after": int(len(boxes))}
    else:
        debug["nested_suppression"] = {"before": int(len(boxes)), "after": int(len(boxes))}

    # ------------------------
    # Lightweight scoring and truncation
    # - persistence: rewards temporal stability
    # - edge-change: rewards structured change vs sparkle
    # ------------------------
    scored = []
    for b in boxes:
        s_p = _box_score_persistence(gen, b) if cfg.use_persistence else 0.0
        s_e = _box_score_edge_change(curr_log, prev_log_aligned, b) if cfg.use_edge_change_score else 0.0
        area = float(b[2] * b[3])
        scored.append((b, s_p, s_e, area))

    # Sort: persistence first, then edge-change, then area
    scored.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)

    boxes_sorted = [t[0] for t in scored[:cfg.max_boxes]]
    debug["scoring"] = {
        "use_persistence": bool(cfg.use_persistence),
        "use_edge_change_score": bool(cfg.use_edge_change_score),
        "kept": int(len(boxes_sorted)),
    }

    return CandidateGenResult(mask=fused, boxes=boxes_sorted, debug=debug)


def draw_overlay(gray_u8: np.ndarray, boxes: List[Tuple[int, int, int, int]], max_draw: int = 50) -> np.ndarray:
    """Visualization helper (grayscale -> BGR with boxes)."""
    g = ensure_gray_u8(gray_u8)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes[:max_draw]):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis
