# preprocessing.py
# 输出两个特征（LoG 和 intensity）输出两个 mask（valid_mask 和 spec_mask）
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any
import cv2
import numpy as np

BlurMode = Literal["none", "gaussian", "gaussian_median"] #这个也许可以删掉
SubtitleMaskMode = Literal["none", "auto_bottom"]
# FeatureMode = Literal["intensity", "LoG"]两个都要用到，不过在不同的地方使用


@dataclass(frozen=True)
class PreprocessConfig:
    # blur (for intensity output; LoG has its own Gaussian)
    blur_mode: BlurMode = "none"
    kernel_size: Tuple[int, int] = (5, 5)
    sigma: float = 1.0
    median_ksize: int = 5

    # optional local contrast (default off: can amplify noise)
    use_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # log-domain compression (photometric robustness)
    # intensity_out = log1p(alpha * I) mapping to 0..255
    use_log_compress: bool = True
    log_compress_alpha: float = 6.0

    # LoG feature (float/32F inside -> abs -> robust normalize to uint8)
    log_sigma: float = 1.2
    log_laplacian_ksize: int = 3   # odd
    log_norm_percentile: float = 99.5  # robust scaling; avoid min-max blowing up noise

    # subtitle masking
    subtitle_mask_mode: SubtitleMaskMode = "auto_bottom"
    subtitle_mask_ratio: float = 0.12

    # auto detection via canny-edge fraction in bottom band
    auto_edge_frac_thresh: float = 0.004
    auto_canny1: int = 50
    auto_canny2: int = 150

    # specular mask (255 normal, 0 specular-like)
    # "sparkle" detection: high V + low S + local bright (top-hat)
    spec_v_high: int = 230
    spec_s_low: int = 70
    spec_use_tophat: bool = True
    spec_tophat_ksize: int = 9         # odd, bigger -> broader glints
    spec_tophat_thresh: int = 18       # 0..255, lower -> more aggressive

    # "blown-out smooth patch" supplement: very high V + low grad + low S
    spec_v_very_high: int = 250
    spec_s_very_low: int = 90
    spec_grad_low: int = 10            # on |sobelx|+|sobely| (0..510 approx)

    # morphology on spec mask candidate
    spec_dilate_ksize: int = 3         # expand highlight regions slightly


@dataclass
class PreprocessResult:
    intensity: np.ndarray       # uint8 HxW (after blur + optional CLAHE + optional log compression)
    log: np.ndarray             # uint8 HxW (LoG abs + robust normalize)
    valid_mask: np.ndarray      # uint8 HxW, 255=valid, 0=invalid
    spec_mask: np.ndarray       # uint8 HxW, 255=normal, 0=specular-like
    debug: Dict[str, Any]

    @property
    def gray(self) -> np.ndarray:
        """Backward-friendly alias if your old code expects result.gray."""
        return self.intensity


def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    """BGR/Gray -> Gray uint8."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _bottom_band_y(gray_h: int, ratio: float) -> Optional[Tuple[int, int]]:
    ratio = float(np.clip(ratio, 0.0, 0.5))
    band_h = int(round(gray_h * ratio))
    if band_h <= 0:
        return None
    return gray_h - band_h, gray_h


def compute_valid_mask(gray_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return valid_mask (255 valid / 0 invalid) and debug."""
    h, w = gray_u8.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    dbg: Dict[str, Any] = {
        "subtitle_mask_mode": cfg.subtitle_mask_mode,
        "subtitle_mask_ratio": cfg.subtitle_mask_ratio,
        "subtitle_detected": False,
        "subtitle_edge_frac": 0.0,
    }

    if cfg.subtitle_mask_mode == "none":
        return mask, dbg

    band = _bottom_band_y(h, cfg.subtitle_mask_ratio)
    if band is None:
        return mask, dbg
    y0, y1 = band

    if cfg.subtitle_mask_mode == "auto_bottom":
        band_img = gray_u8[y0:y1, :]
        edges = cv2.Canny(band_img, cfg.auto_canny1, cfg.auto_canny2)
        frac = float(np.count_nonzero(edges)) / float(edges.size + 1e-6)
        dbg["subtitle_edge_frac"] = frac
        if frac < cfg.auto_edge_frac_thresh:
            return mask, dbg
        dbg["subtitle_detected"] = True

    mask[y0:y1, :] = 0
    return mask, dbg


def _apply_blur(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.blur_mode == "none":
        return gray
    if cfg.blur_mode == "gaussian":
        return cv2.GaussianBlur(gray, cfg.kernel_size, cfg.sigma)
    if cfg.blur_mode == "gaussian_median":
        g = cv2.GaussianBlur(gray, cfg.kernel_size, cfg.sigma)
        return cv2.medianBlur(g, cfg.median_ksize)
    raise ValueError(f"Unknown blur_mode: {cfg.blur_mode}")


def _apply_clahe(gray_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.use_clahe:
        return gray_u8
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip), tileGridSize=cfg.clahe_grid)
    return clahe.apply(gray_u8)


def _log_compress(gray_u8: np.ndarray, alpha: float) -> np.ndarray:
    """log1p(alpha*I) mapped to 0..255; compress highlights, keep dark detail."""
    a = float(max(1e-6, alpha))
    x = gray_u8.astype(np.float32) / 255.0
    y = np.log1p(a * x) / np.log1p(a)
    out = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def _robust_normalize_u8(x: np.ndarray, percentile: float, mask_u8: Optional[np.ndarray] = None) -> np.ndarray:
    """Scale by p-th percentile (robust), avoid per-frame min-max amplifying noise."""
    x = x.astype(np.float32)
    if mask_u8 is not None and mask_u8.shape == x.shape and np.any(mask_u8 > 0):
        vals = x[mask_u8 > 0]
    else:
        vals = x.reshape(-1)

    if vals.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    p = float(np.percentile(vals, float(np.clip(percentile, 80.0, 99.99))))
    p = max(p, 1e-6)
    y = np.clip(x * (255.0 / p), 0.0, 255.0)
    return y.astype(np.uint8)


def compute_log_feature(intensity_u8: np.ndarray, cfg: PreprocessConfig, mask_u8: Optional[np.ndarray] = None) -> np.ndarray:
    """LoG: Gaussian -> Laplacian(CV_32F) -> abs -> robust normalize to uint8."""
    sigma = float(max(0.1, cfg.log_sigma))
    k = int(cfg.log_laplacian_ksize)
    if k % 2 == 0:
        k += 1

    blur = cv2.GaussianBlur(intensity_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=max(1, k))
    log_abs = np.abs(lap)
    log_u8 = _robust_normalize_u8(log_abs, cfg.log_norm_percentile, mask_u8=mask_u8)
    return log_u8


def compute_spec_mask(bgr: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    spec_mask: 255 normal, 0 specular-like
    Strategy:
      A) sparkle: V high + S low + top-hat(V) high (local bright points/lines)
      B) blown-out smooth: V very high + low gradient + low-ish S
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # A) local bright (top-hat)
    if cfg.spec_use_tophat:
        k = int(cfg.spec_tophat_ksize)
        if k % 2 == 0:
            k += 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, ker)
    else:
        tophat = np.zeros_like(v)

    sparkle = (v >= cfg.spec_v_high) & (s <= cfg.spec_s_low) & (tophat >= cfg.spec_tophat_thresh)

    # B) saturated smooth patch supplement
    sx = cv2.Sobel(v, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(v, cv2.CV_16S, 0, 1, ksize=3)
    ax = cv2.convertScaleAbs(sx)
    ay = cv2.convertScaleAbs(sy)
    grad = (ax.astype(np.int16) + ay.astype(np.int16))  # 0..510 approx

    blown = (v >= cfg.spec_v_very_high) & (s <= cfg.spec_s_very_low) & (grad <= cfg.spec_grad_low)

    cand = sparkle | blown

    # expand a bit so water glints edges are also covered
    if cfg.spec_dilate_ksize and cfg.spec_dilate_ksize > 1:
        kd = int(cfg.spec_dilate_ksize)
        ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        cand_u8 = (cand.astype(np.uint8) * 255)
        cand_u8 = cv2.dilate(cand_u8, ker_d, iterations=1)
        cand = cand_u8 > 0

    spec_mask = np.full(v.shape, 255, dtype=np.uint8)
    spec_mask[cand] = 0

    dbg = {
        "sparkle_ratio": float(np.mean(sparkle)),
        "blown_ratio": float(np.mean(blown)),
        "spec_ratio": float(np.mean(spec_mask == 0)),
        "spec_v_high": int(cfg.spec_v_high),
        "spec_s_low": int(cfg.spec_s_low),
        "spec_tophat_thresh": int(cfg.spec_tophat_thresh),
    }
    return spec_mask, dbg


def combine_masks(*masks: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Combine multiple masks with semantics 255=valid, 0=invalid.
    Return None if all inputs are None.
    """
    ms = [m for m in masks if m is not None]
    if not ms:
        return None
    out = ms[0].copy()
    for m in ms[1:]:
        if m.shape != out.shape:
            raise ValueError("Mask shape mismatch in combine_masks")
        out = cv2.bitwise_and(out, m)
    return out


def apply_valid_mask_fill(gray_u8: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Fill invalid pixels (mask==0) with median of valid region to avoid hard edges.
    Works for combined masks too (valid_mask & spec_mask & ...).
    """
    if valid_mask is None:
        return gray_u8
    m = valid_mask.astype(np.uint8)
    if m.shape != gray_u8.shape:
        raise ValueError("valid_mask shape mismatch")
    if np.all(m == 0):
        return gray_u8

    out = gray_u8.copy()
    valid_pixels = out[m > 0]
    fill_val = int(np.median(valid_pixels)) if valid_pixels.size else 0
    out[m == 0] = fill_val
    return out


def preprocess_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    gray = ensure_gray_u8(bgr)

    valid_mask, mask_dbg = compute_valid_mask(gray, cfg)
    spec_mask, spec_dbg = compute_spec_mask(bgr, cfg)

    # intensity pipeline
    inten = _apply_blur(gray, cfg)
    inten = _apply_clahe(inten, cfg)
    if cfg.use_log_compress:
        inten = _log_compress(inten, cfg.log_compress_alpha)

    # LoG computed from intensity (already photometric-robust)
    # Note: normalize using valid_mask only (spec_mask can be very dynamic); you can change to combined if needed.
    log_u8 = compute_log_feature(inten, cfg, mask_u8=valid_mask)

    dbg = {
        "valid_mask": mask_dbg,
        "spec_mask": spec_dbg,
        "use_log_compress": bool(cfg.use_log_compress),
        "log_compress_alpha": float(cfg.log_compress_alpha),
        "use_clahe": bool(cfg.use_clahe),
        "blur_mode": cfg.blur_mode,
        "log_sigma": float(cfg.log_sigma),
    }

    return PreprocessResult(
        intensity=inten,
        log=log_u8,
        valid_mask=valid_mask,
        spec_mask=spec_mask,
        debug=dbg,
    )
