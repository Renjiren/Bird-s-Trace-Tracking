# preprocessing.py
# Step1 outputs:
#   - intensity: I_norm (illumination-normalized, uint8)
#   - intensity_smooth: smoothed gray intensity (uint8)  [Change A]
#   - log: LoG computed on I_norm (uint8)
#   - valid_mask: subtitle mask (glyph-level + safe dilation), computed once and reused in Step2/Step3


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Literal

import cv2
import numpy as np

SmoothMode = Literal["none", "bilateral"]


# Config
@dataclass(frozen=True)
class PreprocessConfig:
    # ---------- denoise for intensity base ----------
    # Purpose: reduce sensor/compression noise, make inorm more stable.
    # Avoid over-smoothing: do NOT aim to "blur the whole background".
    smooth_mode: SmoothMode = "bilateral"
    bilateral_d: int = 7
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 15.0

    # ---------- illumination normalization (I_norm) ----------
    # Inorm_f(x) = log(I(x)+eps) - log(G_sigma * I(x) + eps)
    illum_sigma: float = 25.0
    illum_eps: float = 1.0

    # Map log-domain to uint8:
    # Inorm_u8 = ((clip(Inorm_f, -c, c)/c)*127 + 128)
    norm_clip_abs: float = 1.2

    # ---------- LoG on I_norm (camera motion feature) ----------
    log_blur_sigma: float = 1.2
    log_ksize: int = 3

    # ---------- subtitle ROI ----------
    # IMPORTANT: start from 0.74H to avoid missing subtitles
    subtitle_roi_y0_ratio: float = 0.74
    subtitle_roi_y1_ratio: float = 1.00

    # ---------- subtitle glyph detection (V top-hat + S filter) ----------
    subtitle_delta_sigma: float = 6.0
    subtitle_delta_th: float = 12.0
    subtitle_v_min: int = 160
    subtitle_s_max: int = 70

    # ---------- CC filtering for subtitle line blobs ----------
    subtitle_min_area: int = 80
    subtitle_min_width_ratio: float = 0.12
    subtitle_max_height_ratio: float = 0.20

    # ---------- morphology ----------
    # small close to bridge tiny gaps / anti-aliasing breaks
    subtitle_morph_ksize: Tuple[int, int] = (3, 1)
    # BIGGER "safe boundary" dilation (key for no subtitle residue)
    subtitle_safe_dilate_ksize: Tuple[int, int] = (9, 3)


# Result
@dataclass
class PreprocessResult:
    intensity: np.ndarray          # uint8 HxW, I_norm mapped to [0,255]
    intensity_smooth: np.ndarray   # uint8 HxW, smoothed gray (for Step3 fallback / fusion)
    log: np.ndarray                # uint8 HxW, LoG on I_norm
    valid_mask: np.ndarray         # uint8 HxW, 255 valid / 0 subtitle
    smooth_bgr: np.ndarray         # uint8 HxWx3, optional debug/visualization
    debug: Dict[str, Any]


# Utils
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


# Smoothing (denoise)
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


# Illumination-normalized intensity (I_norm)
def compute_inorm(gray_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    I_norm in log-domain:
      Inorm_f(x) = log(I(x)+eps) - log(G_sigma * I(x) + eps)
    Map to uint8 with stable clipping:
      Inorm_u8 = ((clip(Inorm_f, -c, c)/c)*127 + 128)
    """
    eps = float(max(1e-6, cfg.illum_eps))
    sigma = float(max(0.0, cfg.illum_sigma))
    clip_abs = float(max(1e-6, cfg.norm_clip_abs))

    I = gray_u8.astype(np.float32)

    if sigma > 0:
        illum = cv2.GaussianBlur(I, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        illum = I

    inorm_f = np.log(I + eps) - np.log(illum + eps)
    inorm_f_clip = np.clip(inorm_f, -clip_abs, clip_abs)

    inorm_u8 = ((inorm_f_clip / clip_abs) * 127.0 + 128.0)
    inorm_u8 = np.clip(inorm_u8, 0.0, 255.0).astype(np.uint8)

    dbg = {
        "illum_sigma": sigma,
        "illum_eps": eps,
        "norm_clip_abs": clip_abs,
        "inorm_f_p01": float(np.percentile(inorm_f, 1.0)),
        "inorm_f_p50": float(np.percentile(inorm_f, 50.0)),
        "inorm_f_p99": float(np.percentile(inorm_f, 99.0)),
    }
    return inorm_u8, dbg


# LoG feature (extract background edges: computed on I_norm)
def compute_log_feature(gray_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    LoG pipeline:
      blur = G_{sigma_s} * gray
      lap = |∇^2 blur|
      normalize by robust percentiles -> uint8
    """
    sigma = float(max(0.0, cfg.log_blur_sigma))
    if sigma > 0:
        gray_u8 = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)

    k = int(cfg.log_ksize)
    if k <= 0:
        k = 3
    if k % 2 == 0:
        k += 1

    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=k)
    lap = np.abs(lap)

    # robust normalize to uint8
    p1 = float(np.percentile(lap, 1.0))
    p99 = float(np.percentile(lap, 99.0))
    lap = np.clip((lap - p1) / max(1e-6, (p99 - p1)), 0.0, 1.0)
    return (lap * 255.0).astype(np.uint8)


# Subtitle mask (glyph-level + safe dilation)
def compute_subtitle_valid_mask(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Steps:
      1) ROI: y in [0.74H, 1.0H]
      2) glyph candidate via top-hat on V and low S:
           delta = max(V - G_sigma(V), 0)
           glyph = (V>=v_min) & (delta>=delta_th) & (S<=s_max)
      3) light morphology close to connect anti-aliased glyph pixels
      4) CC filter to keep line-like blobs (wide & low height)
      5) SAFE dilation to cover outlines/shadows -> ensure no subtitle residue

    Output:
      valid_mask: 255 valid / 0 subtitle
    """
    h, w = bgr_u8.shape[:2]
    valid_mask = np.full((h, w), 255, dtype=np.uint8)

    y0 = int(round(h * float(np.clip(cfg.subtitle_roi_y0_ratio, 0.0, 1.0))))
    y1 = int(round(h * float(np.clip(cfg.subtitle_roi_y1_ratio, 0.0, 1.0))))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if y1 <= y0:
        return valid_mask, {"roi": [0, y0, w, y1], "components_kept": 0, "invalid_ratio": 0.0}

    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    # top-hat on V: delta = V - blur(V)
    V_f = V.astype(np.float32)
    sigma = float(max(0.0, cfg.subtitle_delta_sigma))
    V_blur = cv2.GaussianBlur(V_f, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else V_f
    delta = np.maximum(V_f - V_blur, 0.0)

    glyph = (
        (V >= int(cfg.subtitle_v_min))
        & (delta >= float(cfg.subtitle_delta_th))
        & (S <= int(cfg.subtitle_s_max))
    )

    roi = (glyph[y0:y1, :].astype(np.uint8)) * 255

    # light morphology: close to bridge broken strokes / anti-aliasing gaps
    kx, ky = cfg.subtitle_morph_ksize
    kx, ky = max(1, int(kx)), max(1, int(ky))
    if kx > 1 or ky > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, ker)

    # CC filter: keep subtitle-line-like blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)

    min_area = int(cfg.subtitle_min_area)
    min_w = int(round(float(cfg.subtitle_min_width_ratio) * w))
    max_h = int(round(float(cfg.subtitle_max_height_ratio) * h))

    kept = 0
    keep_mask = np.zeros_like(roi)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        if ww < min_w:
            continue
        if hh > max_h:
            continue
        keep_mask[labels == i] = 255
        kept += 1

    # SAFE dilation: cover outlines/shadows/anti-aliasing residues (key)
    if kept > 0:
        sx, sy = cfg.subtitle_safe_dilate_ksize
        sx, sy = max(1, int(sx)), max(1, int(sy))
        ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (sx, sy))
        keep_mask = cv2.dilate(keep_mask, ker2, iterations=1)
        valid_mask[y0:y1, :][keep_mask > 0] = 0

    dbg = {
        "roi": [0, y0, w, y1],
        "delta_sigma": sigma,
        "v_min": int(cfg.subtitle_v_min),
        "delta_th": float(cfg.subtitle_delta_th),
        "s_max": int(cfg.subtitle_s_max),
        "components_total": int(max(0, num - 1)),
        "components_kept": int(kept),
        "invalid_ratio": float(np.mean(valid_mask == 0)),
    }
    return valid_mask, dbg


# Main entry
def preprocess_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    bgr_u8 = ensure_bgr_u8(bgr)

    # 1) mild denoise for stable intensity base
    smooth_bgr = smooth_background(bgr_u8, cfg)
    smooth_gray = ensure_gray_u8(smooth_bgr)

    # output smooth gray as an extra channel for Step3 fusion
    intensity_smooth = smooth_gray

    # 2) illumination-normalized intensity (this replaces old intensity)
    inorm_u8, dbg_inorm = compute_inorm(smooth_gray, cfg)

    # 3) LoG computed on I_norm (for Step2 camera motion compensation)
    log_u8 = compute_log_feature(inorm_u8, cfg)

    # 4) subtitle valid_mask (computed once; Step2/Step3 reuse)
    valid_mask, dbg_sub = compute_subtitle_valid_mask(bgr_u8, cfg)

    debug = {
        "inorm": dbg_inorm,
        "subtitle": dbg_sub,
        "smooth": {
            "mode": cfg.smooth_mode,
            "bilateral_d": int(cfg.bilateral_d),
            "sigmaColor": float(cfg.bilateral_sigma_color),
            "sigmaSpace": float(cfg.bilateral_sigma_space),
        },
        "log": {
            "blur_sigma": float(cfg.log_blur_sigma),
            "ksize": int(cfg.log_ksize),
        },
    }

    return PreprocessResult(
        intensity=inorm_u8,
        intensity_smooth=intensity_smooth,
        log=log_u8,
        valid_mask=valid_mask,
        smooth_bgr=smooth_bgr,
        debug=debug,
    )

