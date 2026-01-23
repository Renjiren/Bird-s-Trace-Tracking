# preprocessing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import cv2
import numpy as np

BlurMode = Literal["none", "gaussian", "gaussian_median"]
SubtitleMaskMode = Literal["none",  "auto_bottom"]

@dataclass(frozen=True)
class PreprocessConfig:
    blur_mode: BlurMode = "none"
    kernel_size: Tuple[int, int] = (5, 5)
    sigma: float = 1.0
    median_ksize: int = 5

    subtitle_mask_mode: SubtitleMaskMode = "auto_bottom"
    subtitle_mask_ratio: float = 0.12  # 屏蔽底部高度比例（0.12≈12%）

    # auto_bottom 的简单判据：底部条带里“高对比边缘像素占比”大于阈值，认为有字幕
    auto_edge_frac_thresh: float = 0.004  # 0.4% edge pixels
    auto_canny1: int = 50
    auto_canny2: int = 150


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """BGR -> Gray uint8"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def gaussian_blur(gray: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, kernel_size, sigma)


def gaussian_median_blur(gray: np.ndarray, kernel_size: Tuple[int, int], sigma: float, m_ksize: int) -> np.ndarray:
    g = cv2.GaussianBlur(gray, kernel_size, sigma)
    return cv2.medianBlur(g, m_ksize)


def _bottom_band(gray: np.ndarray, ratio: float) -> Optional[Tuple[int, int]]:
    h = gray.shape[0]
    ratio = float(np.clip(ratio, 0.0, 0.5))
    band_h = int(round(h * ratio))
    if band_h <= 0:
        return None
    return (h - band_h, h)


def _auto_detect_subtitle(gray: np.ndarray, y0: int, y1: int, canny1: int, canny2: int) -> float:
    band = gray[y0:y1, :]
    edges = cv2.Canny(band, canny1, canny2)
    frac = float(np.count_nonzero(edges)) / float(edges.size + 1e-6)
    return frac


def apply_subtitle_mask(gray: np.ndarray, mode: SubtitleMaskMode, ratio: float,
                        auto_edge_frac_thresh: float, canny1: int, canny2: int) -> np.ndarray:
    if mode == "none":
        return gray

    band = _bottom_band(gray, ratio)
    if band is None:
        return gray
    y0, y1 = band

    if mode == "auto_bottom":
        frac = _auto_detect_subtitle(gray, y0, y1, canny1, canny2)
        if frac < auto_edge_frac_thresh:
            return gray  # 认为没有字幕，不屏蔽

    out = gray.copy()
    out[y0:y1, :] = 0
    return out


def preprocess_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    gray = to_grayscale(bgr)

    if cfg.blur_mode == "gaussian":
        gray = gaussian_blur(gray, cfg.kernel_size, cfg.sigma)
    elif cfg.blur_mode == "gaussian_median":
        gray = gaussian_median_blur(gray, cfg.kernel_size, cfg.sigma, cfg.median_ksize)
    elif cfg.blur_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown blur_mode: {cfg.blur_mode}")

    gray = apply_subtitle_mask(
        gray,
        mode=cfg.subtitle_mask_mode,
        ratio=cfg.subtitle_mask_ratio,
        auto_edge_frac_thresh=cfg.auto_edge_frac_thresh,
        canny1=cfg.auto_canny1,
        canny2=cfg.auto_canny2,
    )
    return gray