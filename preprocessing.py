# preprocessing.py
# Step1: output features（LoG based on local energy normalization & intensity_smooth）+  mask（valid_mask & spec_mask）

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple, Literal, List, Optional
import cv2
import numpy as np

SubtitleMaskMode = Literal["none", "spec_roi"]
SmoothMode = Literal["none", "bilateral"]
SpecEnableMode = Literal["always", "texture_only"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing configuration parameters - fixed subtitle parameters, adaptive glare parameters"""

    # ---------- Subtitle detection configuration (fixed parameters, not involved in adaptive testing) ----------
    # decide the roi for subtitle detection
    subtitle_mask_mode: SubtitleMaskMode = "spec_roi"
    subtitle_roi_y0_ratio: float = 0.74
    subtitle_roi_y1_ratio: float = 1.00

    subtitle_min_area: int = 80
    subtitle_min_width_ratio: float = 0.12
    subtitle_max_height_ratio: float = 0.20
    subtitle_min_y_ratio: float = 0.74
    subtitle_center_x_min: float = 0.03
    subtitle_center_x_max: float = 0.97

    # dilation kernel sizes
    subtitle_group_dilate_ksize: Tuple[int, int] = (25, 3)
    subtitle_glyph_dilate_ksize: Tuple[int, int] = (3, 1)

    # Subtitle spec-like detection parameters
    sub_spec_v_min: int = 180
    sub_spec_delta_sigma: float = 6.0
    sub_spec_delta_th: float = 12.0
    sub_spec_s_low: int = 55
    sub_spec_delta_strong_mul: float = 2.0

    # ---------- Background smooth configuration ----------
    smooth_mode: SmoothMode = "bilateral"
    bilateral_d: int = 7
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 15.0

    # ---------- LoG feature configuration ----------
    log_blur_sigma: float = 1.2
    log_ksize: int = 3
    log_edge_th: int = 35
    use_local_normalization: bool = True  # Local energy normalization

    # ---------- Glare detection basic configuration ----------
    glare_fill_enable: bool = True
    glare_fill_v_offset: int = 12
    glare_fill_delta_th: float = 6.0
    glare_fill_dilate_ksize: int = 11
    glare_fill_iters: int = 1

    glare_cc_min_area: int = 8
    glare_cc_max_area_ratio: float = 0.030
    glare_total_ratio_max: float = 0.080
    glare_max_component_ratio_max: float = 0.050
    glare_cc_edge_density_min: float = 0.010

    # ---------- Glare detection adaptive configuration (adjusted according to video)----------
    spec_enable_mode: SpecEnableMode = "texture_only"
    spec_texture_edge_density_min: float = 0.020

    glare_spec_v_min: int = 230
    glare_spec_delta_sigma: float = 6.0
    glare_spec_delta_th: float = 24.0
    glare_spec_s_low: int = 55
    glare_spec_delta_strong_mul: float = 2.5


@dataclass
class PreprocessResult:
    intensity: np.ndarray     # uint8 HxW, gray image after smooth  (intensity_smooth)
    log: np.ndarray           # uint8 HxW, LoG feature
    valid_mask: np.ndarray    # uint8 HxW, 255 valid / 0 invalid（subtitle）
    spec_mask: np.ndarray     # uint8 HxW, 255 normal / 0 mask（soft mask）
    smooth_bgr: np.ndarray    # uint8 HxWx3, bgr image after smooth
    debug: Dict[str, Any]     # debug info for analysis


def ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def smooth_background(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.smooth_mode == "none":
        return bgr_u8
    return cv2.bilateralFilter(
        bgr_u8,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )


def compute_log_feature(gray_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    LoG features (more robust to continuous illumination changes):

    - Light Gaussian first

    - Laplacian -> abs

    - Choice: Local energy normalization (effective when use_local_normalization=True)

    - Finally, robust percentile mapping to 0..255
    """
    g = gray_u8
    sigma = float(max(0.0, cfg.log_blur_sigma))
    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)

    k = int(cfg.log_ksize)
    if k <= 0:
        k = 3
    if k % 2 == 0:
        k += 1

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=k)
    lap = np.abs(lap)

    if cfg.use_local_normalization:
        # Local energy normalization: Suppressing false edge changes caused by "overall brightening/darkening"
        s = float(max(2.5, 3.0 * sigma + 2.0))
        energy = cv2.GaussianBlur(lap, (0, 0), sigmaX=s, sigmaY=s)
        lap = lap / (energy + 1e-6)

    p1 = float(np.percentile(lap, 1.0))
    p99 = float(np.percentile(lap, 99.0))
    denom = max(1e-6, (p99 - p1))
    norm = (lap - p1) / denom
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def apply_valid_mask_fill(gray_u8: np.ndarray, valid_mask: Optional[np.ndarray], sigma: float = 3.0) -> np.ndarray:
    """
    Hard mask (for invalid subtitles area) "filling in the gaps":

    Replacing invalid areas with blur prevents strong false signals from appearing at subtitle edges due to diff/phase-corr.

    This isn't "strengthening the chain," but rather ensuring subsequent operators see a continuous image.
    """
    g = ensure_gray_u8(gray_u8)
    if valid_mask is None:
        return g
    m = valid_mask.astype(np.uint8) if valid_mask.dtype != np.uint8 else valid_mask
    if m.shape != g.shape or (not np.any(m == 0)):
        return g
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=float(max(0.0, sigma)), sigmaY=float(max(0.0, sigma)))
    out = g.copy()
    out[m == 0] = blur[m == 0]
    return out


def _compute_delta_from_v(V_u8: np.ndarray, sigma: float) -> np.ndarray:
    V_f = V_u8.astype(np.float32)
    s = float(max(0.0, sigma))
    V_blur = cv2.GaussianBlur(V_f, (0, 0), sigmaX=s, sigmaY=s) if s > 0 else V_f
    return np.maximum(V_f - V_blur, 0.0)


def _spec_like_from_svdelta(
    S_u8: np.ndarray,
    V_u8: np.ndarray,
    delta_f32: np.ndarray,
    *,
    v_min: int,
    delta_th: float,
    s_low: int,
    delta_strong_mul: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    v_min_i = int(v_min)
    d_th = float(delta_th)
    d_strong = float(delta_th * max(1.0, delta_strong_mul))

    low_s = (S_u8 <= int(s_low))
    core = (V_u8 >= v_min_i) & (delta_f32 >= d_th)
    spec_like = core & (low_s | (delta_f32 >= d_strong))

    dbg = {
        "v_min": v_min_i,
        "delta_th": d_th,
        "s_low": int(s_low),
        "delta_strong": d_strong,
        "v_p99": float(np.percentile(V_u8.astype(np.float32), 99.0)),
        "delta_p995": float(np.percentile(delta_f32, 99.5)),
        "raw_spec_ratio": float(np.mean(spec_like)),
    }
    return spec_like, dbg


def compute_valid_mask_from_spec_roi(
    sub_spec_like: np.ndarray,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = sub_spec_like.shape[:2]
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    dbg: Dict[str, Any] = {
        "mode": cfg.subtitle_mask_mode,
        "roi": None,
        "components_total": 0,
        "components_kept": 0,
        "invalid_ratio": 0.0,
    }

    if cfg.subtitle_mask_mode == "none":
        return valid_mask, dbg

    y0 = int(round(h * float(np.clip(cfg.subtitle_roi_y0_ratio, 0.0, 1.0))))
    y1 = int(round(h * float(np.clip(cfg.subtitle_roi_y1_ratio, 0.0, 1.0))))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if y1 <= y0:
        return valid_mask, dbg

    dbg["roi"] = [0, y0, w, y1]

    roi = sub_spec_like[y0:y1, :]
    glyph_u8 = (roi.astype(np.uint8) * 255)

    gx, gy = cfg.subtitle_group_dilate_ksize
    gx, gy = max(1, int(gx)), max(1, int(gy))
    group_u8 = glyph_u8
    if gx > 1 or gy > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (gx, gy))
        group_u8 = cv2.dilate(glyph_u8, ker, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(group_u8, connectivity=8)
    dbg["components_total"] = int(max(0, num - 1))

    min_area = int(cfg.subtitle_min_area)
    min_w = int(round(float(cfg.subtitle_min_width_ratio) * w))
    max_h = int(round(float(cfg.subtitle_max_height_ratio) * h))
    min_y = int(round(float(cfg.subtitle_min_y_ratio) * h))
    cx_min = float(cfg.subtitle_center_x_min) * w
    cx_max = float(cfg.subtitle_center_x_max) * w

    keep_group = np.zeros_like(glyph_u8, dtype=np.uint8)
    kept = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        top_y = y0 + y
        cx = float(centroids[i][0])

        if area < min_area:
            continue
        if ww < min_w:
            continue
        if hh > max_h:
            continue
        if top_y < min_y:
            continue
        if not (cx_min <= cx <= cx_max):
            continue

        keep_group[labels == i] = 255
        kept += 1

    dbg["components_kept"] = int(kept)

    if kept > 0:
        kx, ky = cfg.subtitle_glyph_dilate_ksize
        kx, ky = max(1, int(kx)), max(1, int(ky))
        glyph_final = glyph_u8
        if kx > 1 or ky > 1:
            ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
            glyph_final = cv2.dilate(glyph_u8, ker2, iterations=1)

        final_u8 = np.zeros_like(glyph_u8)
        final_u8[(keep_group > 0) & (glyph_final > 0)] = 255

        vm = valid_mask[y0:y1, :]
        vm[final_u8 > 0] = 0
        valid_mask[y0:y1, :] = vm

    dbg["invalid_ratio"] = float(np.mean(valid_mask == 0))
    return valid_mask, dbg


def edge_density(log_u8: np.ndarray, valid_mask: np.ndarray, edge_th: int) -> float:
    valid = (valid_mask > 0)
    if not np.any(valid):
        return 0.0
    return float(np.mean((log_u8 >= int(edge_th)) & valid))


def compute_spec_mask_glare(
    glare_spec_like: np.ndarray,
    valid_mask: np.ndarray,
    log_u8: np.ndarray,
    V_u8: np.ndarray,
    delta_f32: np.ndarray,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    spec detection：
    output spec_mask：255 normal / 0 glare-like
    """
    h, w = glare_spec_like.shape[:2]
    dbg: Dict[str, Any] = {
        "enable_mode": cfg.spec_enable_mode,
        "enabled": True,
        "edge_density": 0.0,
        "fill_used": False,
        "fill_v_min": None,
        "fill_delta_th": None,
        "cc_total": 0,
        "cc_kept": 0,
        "spec_ratio_valid_final": 0.0,
        "fallback": None,
    }

    valid = (valid_mask > 0)

    # texture density check
    ed = edge_density(log_u8, valid_mask, cfg.log_edge_th)
    dbg["edge_density"] = float(ed)

    enabled = True
    if cfg.spec_enable_mode == "texture_only":
        enabled = (ed >= float(cfg.spec_texture_edge_density_min))
    dbg["enabled"] = bool(enabled)

    spec_mask = np.full((h, w), 255, dtype=np.uint8)
    if not enabled:
        return spec_mask, dbg

    seed_u8 = np.zeros((h, w), dtype=np.uint8)
    seed_u8[glare_spec_like & valid] = 255
    glare_u8 = seed_u8

    # V+delta filling
    if cfg.glare_fill_enable and np.any(seed_u8 > 0):
        v_fill_min = max(0, int(cfg.glare_spec_v_min) - int(cfg.glare_fill_v_offset))
        d_fill_th = float(max(0.0, cfg.glare_fill_delta_th))
        dbg["fill_v_min"] = int(v_fill_min)
        dbg["fill_delta_th"] = float(d_fill_th)

        bright = (V_u8 >= v_fill_min) & (delta_f32 >= d_fill_th) & valid

        k = max(1, int(cfg.glare_fill_dilate_ksize))
        if k % 2 == 0:
            k += 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grow = cv2.dilate(seed_u8, ker, iterations=int(max(1, cfg.glare_fill_iters)))

        filled = np.zeros_like(seed_u8)
        filled[(grow > 0) & bright] = 255

        glare_u8 = cv2.bitwise_or(seed_u8, filled)
        dbg["fill_used"] = True

    # CC filtering
    num, labels, stats, _ = cv2.connectedComponentsWithStats(glare_u8, connectivity=8)
    dbg["cc_total"] = int(max(0, num - 1))

    total_px = int(h * w)
    max_area_px = int(round(float(cfg.glare_cc_max_area_ratio) * total_px))
    max_area_px = max(max_area_px, int(cfg.glare_cc_min_area) + 1)
    min_area_px = int(cfg.glare_cc_min_area)

    keep = np.zeros_like(glare_u8)
    kept = 0
    max_comp_area = 0
    sum_area = 0

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area_px:
            continue
        if area > max_area_px:
            continue

        patch_log = log_u8[y:y + hh, x:x + ww]
        patch_valid = valid_mask[y:y + hh, x:x + ww] > 0
        if np.any(patch_valid):
            local_ed = float(np.mean((patch_log >= int(cfg.log_edge_th)) & patch_valid))
            if local_ed < float(cfg.glare_cc_edge_density_min):
                continue

        keep[labels == i] = 255
        kept += 1
        sum_area += int(area)
        max_comp_area = max(max_comp_area, int(area))

    dbg["cc_kept"] = int(kept)

    valid_count = int(np.count_nonzero(valid))
    total_ratio = float(sum_area) / float(valid_count) if valid_count > 0 else 0.0
    max_ratio = float(max_comp_area) / float(valid_count) if valid_count > 0 else 0.0

    if total_ratio > float(cfg.glare_total_ratio_max):
        dbg["fallback"] = {"reason": "total_ratio_too_high", "total_ratio": total_ratio}
        return np.full((h, w), 255, dtype=np.uint8), dbg

    if max_ratio > float(cfg.glare_max_component_ratio_max):
        dbg["fallback"] = {"reason": "max_component_too_high", "max_ratio": max_ratio}
        return np.full((h, w), 255, dtype=np.uint8), dbg

    spec_mask[keep > 0] = 0
    spec_mask[~valid] = 255

    dbg["spec_ratio_valid_final"] = float(np.mean((spec_mask == 0)[valid])) if valid_count > 0 else 0.0
    return spec_mask, dbg


# ---------- Video adaptive: based on fisrt N frames -> replace cfg adaptive ----------
def compute_frame_stats(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> Dict[str, float]:
    h, w = bgr_u8.shape[:2]
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    delta = _compute_delta_from_v(V, cfg.glare_spec_delta_sigma)

    gray_u8 = ensure_gray_u8(bgr_u8)
    log_u8 = compute_log_feature(gray_u8, cfg)

    border = int(min(h, w) * 0.1)
    valid_region = np.zeros((h, w), dtype=bool)
    if h > 2 * border and w > 2 * border:
        valid_region[border:h-border, border:w-border] = True

    V_valid = V[valid_region] if np.any(valid_region) else V
    brightness_mean = float(np.mean(V_valid)) if V_valid.size > 0 else 128.0

    delta_valid = delta[valid_region] if np.any(valid_region) else delta
    delta_p995 = float(np.percentile(delta_valid, 99.5)) if delta_valid.size > 0 else 0.0

    valid_mask_u8 = (valid_region * 255).astype(np.uint8)
    edge_density = edge_density(log_u8, valid_mask_u8, cfg.log_edge_th)

    return {
        "brightness_mean": float(brightness_mean),
        "delta_p995": float(delta_p995),
        "edge_density": float(edge_density),
    }


def adapt_glare_params_for_video(
    stats_list: List[Dict[str, float]],
    base_cfg: PreprocessConfig
) -> Tuple[PreprocessConfig, Dict[str, Any]]:
    if not stats_list:
        return base_cfg, {"used": False, "reason": "no_stats"}

    brightness_med = float(np.median([s["brightness_mean"] for s in stats_list]))
    delta_p995_med = float(np.median([s["delta_p995"] for s in stats_list]))
    edge_density_med = float(np.median([s["edge_density"] for s in stats_list]))

    # brightness adaptive v_min
    if brightness_med < 100:
        new_glare_v_min = max(210, int(brightness_med * 1.8))
    elif brightness_med > 180:
        new_glare_v_min = min(250, int(brightness_med * 1.2))
    else:
        new_glare_v_min = int(brightness_med * 1.4)
    new_glare_v_min = max(200, min(250, new_glare_v_min))

    # contrast adaptive delta_th
    contrast_scale = delta_p995_med / 30.0
    contrast_scale = max(0.5, min(2.0, contrast_scale))
    new_glare_delta_th = float(base_cfg.glare_spec_delta_th * contrast_scale)
    new_glare_delta_th = max(15.0, min(60.0, new_glare_delta_th))

    # texture adaptive edge_density_min & enable_mode
    if edge_density_med < 0.01:
        new_texture_min = 0.005
        new_enable_mode: SpecEnableMode = "always"
    elif edge_density_med > 0.08:
        new_texture_min = edge_density_med * 0.4
        new_enable_mode = "texture_only"
    else:
        new_texture_min = edge_density_med * 0.6
        new_enable_mode = "texture_only"

    new_texture_min = max(0.005, min(0.08, float(new_texture_min)))

    adapted_cfg = replace(
        base_cfg,
        spec_enable_mode=new_enable_mode,
        spec_texture_edge_density_min=new_texture_min,
        glare_spec_v_min=new_glare_v_min,
        glare_spec_delta_th=new_glare_delta_th,
    )

    debug_info = {
        "used": True,
        "brightness_med": brightness_med,
        "delta_p995_med": delta_p995_med,
        "edge_density_med": edge_density_med,
        "glare_spec_v_min": int(new_glare_v_min),
        "glare_spec_delta_th": float(new_glare_delta_th),
        "spec_texture_edge_density_min": float(new_texture_min),
        "spec_enable_mode": new_enable_mode,
    }
    return adapted_cfg, debug_info


def preprocess_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    bgr_u8 = ensure_bgr_u8(bgr)

    gray_u8 = ensure_gray_u8(bgr_u8)
    smooth_bgr = smooth_background(bgr_u8, cfg)
    smooth_gray = ensure_gray_u8(smooth_bgr)

    log_u8 = compute_log_feature(gray_u8, cfg)

    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, S_u8, V_u8 = cv2.split(hsv)

    # subtitle（fixied paramaters）
    sub_delta = _compute_delta_from_v(V_u8, cfg.sub_spec_delta_sigma)
    sub_spec_like, dbg_sub_spec = _spec_like_from_svdelta(
        S_u8, V_u8, sub_delta,
        v_min=cfg.sub_spec_v_min,
        delta_th=cfg.sub_spec_delta_th,
        s_low=cfg.sub_spec_s_low,
        delta_strong_mul=cfg.sub_spec_delta_strong_mul,
    )
    dbg_sub_spec["delta_sigma"] = float(cfg.sub_spec_delta_sigma)
    valid_mask, dbg_valid = compute_valid_mask_from_spec_roi(sub_spec_like, cfg)

    # glare（adaptive paramaters）
    glare_delta = _compute_delta_from_v(V_u8, cfg.glare_spec_delta_sigma)
    glare_spec_like, dbg_glare_spec_base = _spec_like_from_svdelta(
        S_u8, V_u8, glare_delta,
        v_min=cfg.glare_spec_v_min,
        delta_th=cfg.glare_spec_delta_th,
        s_low=cfg.glare_spec_s_low,
        delta_strong_mul=cfg.glare_spec_delta_strong_mul,
    )
    dbg_glare_spec_base["delta_sigma"] = float(cfg.glare_spec_delta_sigma)

    spec_mask, dbg_spec_glare = compute_spec_mask_glare(
        glare_spec_like, valid_mask, log_u8, V_u8, glare_delta, cfg
    )

    dbg = {
        "valid": dbg_valid,
        "subtitle_spec": dbg_sub_spec,
        "glare_spec_base": dbg_glare_spec_base,
        "glare_spec": dbg_spec_glare,
        "config_info": {
            "subtitle": {
                "v_min": cfg.sub_spec_v_min,
                "delta_th": cfg.sub_spec_delta_th,
                "delta_sigma": cfg.sub_spec_delta_sigma,
                "s_low": cfg.sub_spec_s_low,
                "fixed": True,
            },
            "glare": {
                "v_min": cfg.glare_spec_v_min,
                "delta_th": cfg.glare_spec_delta_th,
                "delta_sigma": cfg.glare_spec_delta_sigma,
                "texture_min": cfg.spec_texture_edge_density_min,
                "enable_mode": cfg.spec_enable_mode,
                "adaptive": True,
            },
            "log": {
                "use_local_normalization": bool(cfg.use_local_normalization),
                "blur_sigma": float(cfg.log_blur_sigma),
            }
        }
    }

    return PreprocessResult(
        intensity=smooth_gray,
        log=log_u8,
        valid_mask=valid_mask,
        spec_mask=spec_mask,
        smooth_bgr=smooth_bgr,
        debug=dbg,
    )

