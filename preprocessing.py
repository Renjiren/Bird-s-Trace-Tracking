# preprocessing.py
# Step1: 输出两个特征（LoG 和 intensity）+ 两个 mask（valid_mask / spec_mask）
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal

import cv2
import numpy as np

SubtitleMaskMode = Literal["none", "auto_bottom"]
SmoothMode = Literal["none", "bilateral"]


@dataclass(frozen=True)
class PreprocessConfig:
    # ---------- subtitle / invalid region mask ----------
    # 默认只做 auto（不强制整条底边），并尽量把字幕 mask 限定在“底部居中”的真实字幕区域。
    subtitle_mask_mode: SubtitleMaskMode = "auto_bottom"
    subtitle_mask_ratio: float = 0.14  # 底部候选条带高度（比例）
    auto_canny1: int = 60
    auto_canny2: int = 180

    # 先粗判是否可能有字幕：底部条带边缘占比（避免“无字幕帧”误屏蔽）
    auto_edge_frac_thresh: float = 0.0035

    # 字幕区域几何约束（避免 mask 过宽）
    subtitle_center_tol_frac: float = 0.38
    subtitle_min_w_frac: float = 0.14
    subtitle_max_w_frac: float = 0.90
    subtitle_min_h_frac: float = 0.015
    subtitle_max_h_frac: float = 0.18
    subtitle_min_edge_density_in_box: float = 0.020

    # 形态学连接（把文字零碎边缘连成一块）
    subtitle_close_w_frac: float = 0.06
    subtitle_close_h_frac: float = 0.010
    subtitle_dilate_iter: int = 1

    # bbox padding（让 mask 稍微盖住字幕描边）
    subtitle_pad_x_frac: float = 0.06
    subtitle_pad_y_frac: float = 0.25

    # 可选：由 pipeline 在“视频级”提前估计出的固定字幕 bbox（x0,y0,x1,y1）
    subtitle_roi: Optional[Tuple[int, int, int, int]] = None

    # ---------- background suppression ----------
    smooth_mode: SmoothMode = "bilateral"
    bilateral_d: int = 7
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 15.0

    # ---------- LoG feature ----------
    log_blur_sigma: float = 1.2
    log_ksize: int = 3
    log_norm_clip: Tuple[int, int] = (0, 255)
    log_percentile_stride: int = 4  # 计算 p1/p99 时下采样步长（加速）

    # ---------- specular (highlight) mask ----------
    # 主力规则：delta = V - GaussianBlur(V)
    spec_blur_sigma: float = 3.0
    spec_v_min: int = 220
    spec_delta_th: int = 18

    # 辅助规则：V 高且 S 低
    spec_v_high: int = 235
    spec_s_low: int = 55

    # 兜底：极亮且局部梯度低（平坦亮块）
    spec_v_very_high: int = 245
    spec_grad_low: float = 8.0
    spec_grad_ksize: int = 3

    # spec mask 后处理：去掉单像素噪声
    spec_open_ksize: int = 3


@dataclass
class PreprocessResult:
    intensity: np.ndarray     # uint8 HxW
    log: np.ndarray           # uint8 HxW
    valid_mask: np.ndarray    # uint8 HxW (255 valid / 0 invalid)
    spec_mask: np.ndarray     # uint8 HxW (255 normal / 0 spec-like)
    smooth_bgr: np.ndarray    # uint8 HxWx3
    debug: Dict[str, Any]


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


def _bottom_band_y(h: int, ratio: float) -> Optional[Tuple[int, int]]:
    ratio = float(np.clip(ratio, 0.0, 0.5))
    band_h = int(round(h * ratio))
    if band_h <= 0:
        return None
    return h - band_h, h


def detect_subtitle_bbox(gray_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    在底部条带内自动检测字幕 bbox（尽量只遮字幕而不是整条底边）。
    返回 bbox(x0,y0,x1,y1)（全图坐标）或 None。
    """
    h, w = gray_u8.shape[:2]
    dbg: Dict[str, Any] = {
        "mode": cfg.subtitle_mask_mode,
        "ratio": float(cfg.subtitle_mask_ratio),
        "edge_frac": 0.0,
        "bbox": None,
        "picked_score": 0.0,
    }

    if cfg.subtitle_mask_mode == "none":
        return None, dbg

    band = _bottom_band_y(h, cfg.subtitle_mask_ratio)
    if band is None:
        return None, dbg
    y0, y1 = band

    band_img = gray_u8[y0:y1, :]
    edges = cv2.Canny(band_img, int(cfg.auto_canny1), int(cfg.auto_canny2))
    edge_frac = float(np.count_nonzero(edges)) / float(edges.size + 1e-6)
    dbg["edge_frac"] = edge_frac
    if edge_frac < float(cfg.auto_edge_frac_thresh):
        return None, dbg

    # 连接文字边缘：close -> dilate
    close_w = max(9, int(round(w * float(cfg.subtitle_close_w_frac))))
    close_h = max(3, int(round(h * float(cfg.subtitle_close_h_frac))))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    blobs = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    if int(cfg.subtitle_dilate_iter) > 0:
        kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blobs = cv2.dilate(blobs, kernel_d, iterations=int(cfg.subtitle_dilate_iter))

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_w = float(cfg.subtitle_min_w_frac) * w
    max_w = float(cfg.subtitle_max_w_frac) * w
    min_h = float(cfg.subtitle_min_h_frac) * h
    max_h = float(cfg.subtitle_max_h_frac) * h
    center_tol = float(cfg.subtitle_center_tol_frac) * w

    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        if not (min_w <= bw <= max_w):
            continue
        if not (min_h <= bh <= max_h):
            continue

        cx = x + 0.5 * bw
        if abs(cx - 0.5 * w) > center_tol:
            continue

        roi_edges = edges[y:y + bh, x:x + bw]
        dens = float(np.count_nonzero(roi_edges)) / float(roi_edges.size + 1e-6)
        if dens < float(cfg.subtitle_min_edge_density_in_box):
            continue

        score = float(np.count_nonzero(roi_edges))
        candidates.append((score, x, y, bw, bh, dens))

    if not candidates:
        return None, dbg

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, x, y, bw, bh, dens = candidates[0]
    dbg["picked_score"] = float(score)

    pad_x = int(round(float(cfg.subtitle_pad_x_frac) * bw))
    pad_y = int(round(float(cfg.subtitle_pad_y_frac) * bh))
    x0b = max(0, x - pad_x)
    y0b = max(0, y - pad_y)
    x1b = min(w, x + bw + pad_x)
    y1b = min(y1 - y0, y + bh + pad_y)

    bbox = (int(x0b), int(y0 + y0b), int(x1b), int(y0 + y1b))
    dbg["bbox"] = bbox
    dbg["edge_density_in_box"] = float(dens)
    return bbox, dbg


def compute_valid_mask(gray_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    255 = valid, 0 = invalid (subtitle/border). Does not modify image itself.
    """
    h, w = gray_u8.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)

    if cfg.subtitle_mask_mode == "none":
        return mask, {"subtitle_masked": False, "reason": "mode_none"}

    # 视频级固化 ROI（仅当 pipeline 判定字幕常驻时设置）
    if cfg.subtitle_roi is not None:
        x0, y0, x1, y1 = cfg.subtitle_roi
        x0 = int(np.clip(x0, 0, w))
        x1 = int(np.clip(x1, 0, w))
        y0 = int(np.clip(y0, 0, h))
        y1 = int(np.clip(y1, 0, h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
            return mask, {"subtitle_masked": True, "reason": "video_level_roi", "bbox": (x0, y0, x1, y1)}
        return mask, {"subtitle_masked": False, "reason": "video_level_roi_invalid", "bbox": (x0, y0, x1, y1)}

    bbox, dbg = detect_subtitle_bbox(gray_u8, cfg)
    if bbox is None:
        dbg["subtitle_masked"] = False
        return mask, dbg

    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 0
    dbg["subtitle_masked"] = True
    return mask, dbg


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

    stride = int(max(1, cfg.log_percentile_stride))
    sample = lap[::stride, ::stride] if stride > 1 else lap
    p1 = float(np.percentile(sample, 1.0))
    p99 = float(np.percentile(sample, 99.0))
    denom = max(1e-6, (p99 - p1))
    norm = (lap - p1) / denom
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _grad_mag_u8(gray_u8: np.ndarray, ksize: int) -> np.ndarray:
    k = int(max(1, ksize))
    if k % 2 == 0:
        k += 1
    sx = cv2.Sobel(gray_u8, cv2.CV_16S, 1, 0, ksize=k)
    sy = cv2.Sobel(gray_u8, cv2.CV_16S, 0, 1, ksize=k)
    return (cv2.convertScaleAbs(sx) // 2 + cv2.convertScaleAbs(sy) // 2).astype(np.uint8)


def compute_spec_mask(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    255 = normal, 0 = specular-like.
    主力：V & (V - blur(V))。辅助：V 高 & S 低；兜底：极亮且梯度低。
    """
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    sigma = float(max(0.0, cfg.spec_blur_sigma))
    V_blur = cv2.GaussianBlur(V, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else V
    delta = cv2.subtract(V, V_blur)  # uint8, saturate

    v_min = int(cfg.spec_v_min)
    d_th = int(cfg.spec_delta_th)
    rule_main = (V >= v_min) & (delta >= d_th)
    rule_aux = (V >= int(cfg.spec_v_high)) & (S <= int(cfg.spec_s_low))

    mag = _grad_mag_u8(V, int(cfg.spec_grad_ksize))
    rule_flat = (V >= int(cfg.spec_v_very_high)) & (mag <= float(cfg.spec_grad_low))

    spec_like = rule_main | rule_aux | rule_flat
    spec_u8 = np.where(spec_like, 0, 255).astype(np.uint8)

    k = int(cfg.spec_open_ksize)
    if k > 1:
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        spec_u8 = cv2.morphologyEx(spec_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    dbg = {
        "spec_ratio": float(np.mean(spec_u8 == 0)),
        "v_min": int(v_min),
        "delta_th": int(d_th),
        "rule_main_ratio": float(np.mean(rule_main)),
        "rule_aux_ratio": float(np.mean(rule_aux)),
        "rule_flat_ratio": float(np.mean(rule_flat)),
        "blur_sigma": float(sigma),
    }
    return spec_u8, dbg


def preprocess_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    bgr_u8 = ensure_bgr_u8(bgr)
    gray_u8 = ensure_gray_u8(bgr_u8)

    valid_mask, dbg_valid = compute_valid_mask(gray_u8, cfg)

    smooth_bgr = smooth_background(bgr_u8, cfg)
    smooth_gray = ensure_gray_u8(smooth_bgr)

    log_u8 = compute_log_feature(smooth_gray, cfg)
    spec_mask, dbg_spec = compute_spec_mask(bgr_u8, cfg)

    dbg = {
        "valid": dbg_valid,
        "spec": dbg_spec,
        "smooth": {
            "mode": cfg.smooth_mode,
            "bilateral": {
                "d": int(cfg.bilateral_d),
                "sigmaColor": float(cfg.bilateral_sigma_color),
                "sigmaSpace": float(cfg.bilateral_sigma_space),
            },
        },
        "log": {
            "sigma": float(cfg.log_blur_sigma),
            "ksize": int(cfg.log_ksize),
            "percentile_stride": int(cfg.log_percentile_stride),
        },
    }

    return PreprocessResult(
        intensity=gray_u8,
        log=log_u8,
        valid_mask=valid_mask,
        spec_mask=spec_mask,
        smooth_bgr=smooth_bgr,
        debug=dbg,
    )

