# preprocessing.py
# Step1: output two features（LoG and intensity）+ two masks（valid_mask and spec_mask）
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple, Literal, List

import cv2
import numpy as np

SubtitleMaskMode = Literal["none", "spec_roi"]       # use subtitle spec-like find subtitoe in ROI
SmoothMode = Literal["none", "bilateral"]
SpecEnableMode = Literal["always", "texture_only"]   # glare spec enable mode (diferent from subtitle)


@dataclass(frozen=True)
class PreprocessConfig:
    # ---------- subtitle ROI ----------
    subtitle_mask_mode: SubtitleMaskMode = "spec_roi"

    # subtitle hight = 0.74
    subtitle_roi_y0_ratio: float = 0.74
    subtitle_roi_y1_ratio: float = 1.00

    # Filter subtitles by connected components: width, height, and bottom (for "grouped blocks").
    subtitle_min_area: int = 80
    subtitle_min_width_ratio: float = 0.12
    subtitle_max_height_ratio: float = 0.20
    subtitle_min_y_ratio: float = 0.74
    subtitle_center_x_min: float = 0.03
    subtitle_center_x_max: float = 0.97

    # two-stage dilation sizes
    # (A) group：
    # Strong dilation is only used for connected component filtering to "group a line of text into a block" 
    # (not for final masking).
    subtitle_group_dilate_ksize: Tuple[int, int] = (25, 3)
    # (B) glyph：The final result is invalid, which only covers the shape of the character
    # (lightly fills in the broken strokes, without erasing the gaps between the characters).
    subtitle_glyph_dilate_ksize: Tuple[int, int] = (3, 1)

    # ---------- subtitle spec-like params (fixed,no apdative) ----------
    # top-hat style: delta = V - blur(V)
    sub_spec_v_min: int = 180
    sub_spec_delta_sigma: float = 6.0
    sub_spec_delta_th: float = 12.0
    sub_spec_s_low: int = 55
    sub_spec_delta_strong_mul: float = 2.0

    # ---------- background suppression ----------
    smooth_mode: SmoothMode = "bilateral"
    bilateral_d: int = 7
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 15.0

    # ---------- LoG feature ----------
    log_blur_sigma: float = 1.2
    log_ksize: int = 3
    log_edge_th: int = 35  # edge_density statistical threshold

# ---------- glare spec params (allow apdative per vedio) ----------
    glare_spec_v_min: int = 230
    glare_spec_delta_sigma: float = 6.0
    glare_spec_delta_th: float = 24.0
    glare_spec_s_low: int = 55
    glare_spec_delta_strong_mul: float = 2.0

    # glare fill： seed→grow
    glare_fill_enable: bool = True
    glare_fill_v_offset: int = 12          # v_fill_min = glare_v_min - offset
    glare_fill_dilate_ksize: int = 11      # seed → grow kernel size (odd)
    glare_fill_iters: int = 1

    # ---------- glare gating + CC + fallback ----------
    spec_enable_mode: SpecEnableMode = "texture_only"
    spec_texture_edge_density_min: float = 0.020

    glare_cc_min_area: int = 5                # Minimum size of a single block >= 5 px
    glare_cc_max_area_ratio: float = 0.05     # Maximum single block size <= 5% of total image pixels
    glare_total_ratio_max: float = 0.015      # Total spec > 15% -> Turn off Glare (believes it's broken)
    glare_max_component_ratio_max: float = 0.050  # Maximum block > 5% -> Turn off glare

    # Local texture filtering: Prevents accidental retention of large areas of sky/smooth surfaces.
    glare_cc_edge_density_min: float = 0.010


@dataclass
class PreprocessResult:
    intensity: np.ndarray     # uint8 HxW, smooth gray
    log: np.ndarray           # uint8 HxW
    valid_mask: np.ndarray    # uint8 HxW, 255 valid / 0 invalid
    spec_mask: np.ndarray     # uint8 HxW, 255 normal / 0 glare-like (soft suppression)
    smooth_bgr: np.ndarray    # uint8 HxWx3
    debug: Dict[str, Any]

# ---------------- utils -----------------
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


def odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k + (k % 2 == 0)

def scaled_k(base: int, H: int, W: int, ref: int) -> int:
    s = float(min(H, W)) / float(max(1, ref))
    return odd(max(1, int(round(base * s))))


def apply_valid_mask_fill(gray_u8: np.ndarray, valid_mask: Optional[np.ndarray], sigma: float = 3.0) -> np.ndarray:
    """
    Fill invalid areas (valid_mask=0) by Gaussian blur.
    valid_mask: 255 valid / 0 invalid
    """
    if valid_mask is None:
        return gray_u8
    m = valid_mask.astype(np.uint8)
    if not np.any(m == 0):
        return gray_u8
    blur = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=float(max(0.0, sigma)), sigmaY=float(max(0.0, sigma)))
    out = gray_u8.copy()
    out[m == 0] = blur[m == 0]
    return out


# ---------------- core processing functions -----------------
def smooth_background(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.smooth_mode == "none":
        return bgr_u8
    if cfg.smooth_mode == "bilateral":
        d = int(max(1, cfg.bilateral_d))
        return cv2.bilateralFilter(
            bgr_u8,
            d=d,
            sigmaColor=float(cfg.bilateral_sigma_color),
            sigmaSpace=float(cfg.bilateral_sigma_space),
        )
    raise ValueError(f"Unknown smooth_mode: {cfg.smooth_mode}")


def compute_log_feature(gray_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    sigma = float(max(0.0, cfg.log_blur_sigma))
    blur = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else gray_u8

    k = int(cfg.log_ksize)
    if k <= 0:
        k = 3
    if k % 2 == 0:
        k += 1

    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=k)
    lap = np.abs(lap)

    p1 = float(np.percentile(lap, 1.0))
    p99 = float(np.percentile(lap, 99.0))
    denom = max(1e-6, (p99 - p1))
    norm = (lap - p1) / denom
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def spec_like_tophat(
    bgr_u8: np.ndarray,
    *,
    v_min: int,
    delta_sigma: float,
    delta_th: float,
    s_low: int,
    delta_strong_mul: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    top-hat: delta = V - blur(V)
    spec_like = (V >= v_min) & (delta >= delta_th) & (S is low or delta is strong)
    """
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    V_f = V.astype(np.float32)
    sigma = float(max(0.0, delta_sigma))
    V_blur = cv2.GaussianBlur(V_f, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else V_f
    delta = np.maximum(V_f - V_blur, 0.0)

    v_min_i = int(v_min)
    d_th = float(delta_th)
    d_strong = float(delta_th * max(1.0, delta_strong_mul))

    low_s = (S <= int(s_low))
    core = (V >= v_min_i) & (delta >= d_th)
    spec_like = core & (low_s | (delta >= d_strong))

    dbg = {
        "v_min": v_min_i,
        "delta_sigma": sigma,
        "delta_th": d_th,
        "s_low": int(s_low),
        "delta_strong": d_strong,
        "v_p99": float(np.percentile(V_f, 99.0)),
        "delta_p995": float(np.percentile(delta, 99.5)),
        "raw_spec_ratio": float(np.mean(spec_like)),
    }
    return spec_like, dbg


def compute_valid_mask_from_spec_roi(
    sub_spec_like: np.ndarray,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    subtitle mask ：
    - in ROI, fisrtly use sub_spec_like to get character highlighting”
    - (A) group：Strong dilation connects the entire line of subtitles into a single block -> Connectivity filtering uses "subtitle blocks".
    - (B) glyph：Ultimately, invalid only applies to character pixels (slightly filling in broken strokes without obscuring character gaps).
    """
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

    # (A) Grouped Expansion: Used only to connect entire lines of text into blocks, facilitating connected component filtering.
    gx, gy = cfg.subtitle_group_dilate_ksize
    gx, gy = max(1, int(gx)), max(1, int(gy))
    group_u8 = glyph_u8
    if gx > 1 or gy > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (gx, gy))
        group_u8 = cv2.dilate(glyph_u8, ker, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(group_u8, connectivity=8)
    dbg["components_total"] = int(max(0, num - 1))

    # Connected domain filtering for "subtitle blocks"
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
        # (B) Lightly fill in the gaps between characters: only fill in the broken strokes, do not erase the gaps between characters.
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
    bgr_u8: np.ndarray,
    glare_spec_like: np.ndarray,
    valid_mask: np.ndarray,
    log_u8: np.ndarray,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    spec_mask for glare suppression:
    1) gating by texture (edge density)
    2) seed from strong glare-like spec areas
    3) seed → grow fill
    4) CC filtering (area + local texture)
    5) fallback shutoff
    6) output spec_mask (glare-like = 0)
    """
    h, w = glare_spec_like.shape[:2]
    dbg: Dict[str, Any] = {
        "enable_mode": cfg.spec_enable_mode,
        "enabled": True,
        "edge_density": 0.0,
        "fill_used": False,
        "cc_total": 0,
        "cc_kept": 0,
        "spec_ratio_valid_final": 0.0,
        "fallback": None,
    }

    valid = (valid_mask > 0)

    # 1) texture gating: closed in non-texture scenes (to prevent entire sky/bird from being red)
    ed = edge_density(log_u8, valid_mask, cfg.log_edge_th)
    dbg["edge_density"] = float(ed)

    enabled = True
    if cfg.spec_enable_mode == "texture_only":
        enabled = (ed >= float(cfg.spec_texture_edge_density_min))
    dbg["enabled"] = bool(enabled)

    spec_mask = np.full((h, w), 255, dtype=np.uint8)
    if not enabled:
        return spec_mask, dbg

    # 2) seed：glare-like strong spec areas
    seed_u8 = np.zeros((h, w), dtype=np.uint8)
    seed_u8[glare_spec_like & valid] = 255

    glare_u8 = seed_u8

    # 3) seed→grow fill： from strong edges to the interior of the glare spot
    if cfg.glare_fill_enable and np.any(seed_u8 > 0):
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
        _, _, V = cv2.split(hsv)

        v_fill_min = max(0, int(cfg.glare_spec_v_min) - int(cfg.glare_fill_v_offset))
        bright = ((V >= v_fill_min) & valid)

        k = max(1, int(cfg.glare_fill_dilate_ksize))
        if k % 2 == 0:
            k += 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grow = cv2.dilate(seed_u8, ker, iterations=int(max(1, cfg.glare_fill_iters)))

        filled = np.zeros_like(seed_u8)
        filled[(grow > 0) & bright] = 255

        glare_u8 = cv2.bitwise_or(seed_u8, filled)
        dbg["fill_used"] = True

    # 4) CC filtering
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

        # partial local texture check
        patch_log = log_u8[y:y+hh, x:x+ww]
        patch_valid = valid_mask[y:y+hh, x:x+ww] > 0
        if np.any(patch_valid):
            local_ed = float(np.mean((patch_log >= int(cfg.log_edge_th)) & patch_valid))
            if local_ed < float(cfg.glare_cc_edge_density_min):
                continue

        keep[labels == i] = 255
        kept += 1
        sum_area += int(area)
        max_comp_area = max(max_comp_area, int(area))

    dbg["cc_kept"] = int(kept)


    # 5) fallback shutoff: total_ratio high => shutoff only if it looks like "big-area failure"
    valid_count = int(np.count_nonzero(valid))
    total_ratio = float(sum_area) / float(valid_count) if valid_count > 0 else 0.0
    max_ratio   = float(max_comp_area) / float(valid_count) if valid_count > 0 else 0.0
    largest_share = float(max_comp_area) / sum_area if sum_area > 0 else 0.0

    dbg.update(total_ratio=total_ratio, max_ratio=max_ratio, largest_share=largest_share)

    # hard shutoff: single huge component
    if max_ratio > float(cfg.glare_max_component_ratio_max):
        dbg["fallback"] = {"reason": "max_component_too_high", "max_ratio": max_ratio}
        return np.full((h, w), 255, dtype=np.uint8), dbg

    # soft shutoff: total too high + (area concentrated OR too few comps OR mid-size blob)
    if total_ratio > float(cfg.glare_total_ratio_max):
        looks_like_failure = (largest_share >= 0.30) or (kept < 10) or (max_ratio >= 0.01)
        if looks_like_failure:
            dbg["fallback"] = {
                "reason": "total_ratio_high_and_bigblock",
                "total_ratio": total_ratio,
                "largest_share": largest_share,
                "cc_kept": int(kept),
                "max_ratio": max_ratio,
            }
            return np.full((h, w), 255, dtype=np.uint8), dbg


    # 6) outpit spec_mask
    spec_mask[keep > 0] = 0

    # subtitle invalid area set to normal
    spec_mask[~valid] = 255

    dbg["spec_ratio_valid_final"] = float(np.mean((spec_mask == 0)[valid])) if valid_count > 0 else 0.0
    return spec_mask, dbg


# ---------- adaptive：only in glare，no subtitle ----------
def glare_stats_one_frame(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> Dict[str, float]:
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, _, V = cv2.split(hsv)
    V_f = V.astype(np.float32)

    sigma = float(max(0.0, cfg.glare_spec_delta_sigma))
    V_blur = cv2.GaussianBlur(V_f, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else V_f
    delta = np.maximum(V_f - V_blur, 0.0)

    return {
        "v_p99": float(np.percentile(V_f, 99.0)),
        "delta_p995": float(np.percentile(delta, 99.5)),
    }


def adapt_glare_params_from_stats(
    stats_list: List[Dict[str, float]],
    base_cfg: PreprocessConfig,
    edge_density_med: float,
) -> Tuple[PreprocessConfig, Dict[str, Any]]:
    """
    output：per-video cfg） + debug
    """
    if not stats_list:
        return base_cfg, {"used": False}

    v_p99 = float(np.median([s["v_p99"] for s in stats_list]))
    d_p995 = float(np.median([s["delta_p995"] for s in stats_list]))

    # v_min： slightly lower than p99
    new_v_min = int(np.clip(round(v_p99 - 5.0), 210, 245))

    # delta_th： based on p99.5
    new_delta_th = float(np.clip(d_p995 * 0.45, 12.0, 60.0))

    # gating texture threshold adaptive
    new_texture_min = float(np.clip(edge_density_med * 0.65, 0.010, 0.060))

    cfg2 = replace(
        base_cfg,
        glare_spec_v_min=new_v_min,
        glare_spec_delta_th=new_delta_th,
        spec_texture_edge_density_min=new_texture_min,
    )

    dbg = {
        "used": True,
        "v_p99_med": v_p99,
        "delta_p995_med": d_p995,
        "edge_density_med": float(edge_density_med),
        "glare_spec_v_min": new_v_min,
        "glare_spec_delta_th": new_delta_th,
        "spec_texture_edge_density_min": new_texture_min,
    }
    return cfg2, dbg


def preprocess_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    bgr_u8 = ensure_bgr_u8(bgr)

    # smooth -> intensity/log
    smooth_bgr = smooth_background(bgr_u8, cfg)
    smooth_gray = ensure_gray_u8(smooth_bgr)
    log_u8 = compute_log_feature(smooth_gray, cfg)

    # 1) subtitle special spec-(fixed params)
    sub_spec_like, dbg_sub_spec = spec_like_tophat(
        bgr_u8,
        v_min=cfg.sub_spec_v_min,
        delta_sigma=cfg.sub_spec_delta_sigma,
        delta_th=cfg.sub_spec_delta_th,
        s_low=cfg.sub_spec_s_low,
        delta_strong_mul=cfg.sub_spec_delta_strong_mul,
    )

    # 2) valid_mask
    valid_mask, dbg_valid = compute_valid_mask_from_spec_roi(sub_spec_like, cfg)

    # 3) glare spcieal spec（adaptive params）
    glare_spec_like, dbg_glare_spec_base = spec_like_tophat(
        bgr_u8,
        v_min=cfg.glare_spec_v_min,
        delta_sigma=cfg.glare_spec_delta_sigma,
        delta_th=cfg.glare_spec_delta_th,
        s_low=cfg.glare_spec_s_low,
        delta_strong_mul=cfg.glare_spec_delta_strong_mul,
    )

    # 4) glare spec_mask：gating + fill + CC + fallback + invalid
    spec_mask, dbg_spec_glare = compute_spec_mask_glare(
        bgr_u8, glare_spec_like, valid_mask, log_u8, cfg
    )

    dbg = {
        "valid": dbg_valid,
        "subtitle_spec": dbg_sub_spec,
        "glare_spec_base": dbg_glare_spec_base,
        "glare_spec": dbg_spec_glare,
        "smooth": {
            "mode": cfg.smooth_mode,
            "bilateral_d": int(cfg.bilateral_d),
            "sigmaColor": float(cfg.bilateral_sigma_color),
            "sigmaSpace": float(cfg.bilateral_sigma_space),
        },
        "log": {
            "sigma": float(cfg.log_blur_sigma),
            "ksize": int(cfg.log_ksize),
            "edge_th": int(cfg.log_edge_th),
        },
    }

    return PreprocessResult(
        intensity=smooth_gray,
        log=log_u8,
        valid_mask=valid_mask,
        spec_mask=spec_mask,
        smooth_bgr=smooth_bgr,
        debug=dbg,
    )