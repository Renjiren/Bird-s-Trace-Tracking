# preprocessing.py
# Step1: 输出两个特征（LoG 和 intensity）+ 两个 mask（valid_mask / spec_mask）
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple, Literal, List

import cv2
import numpy as np

SubtitleMaskMode = Literal["none", "spec_roi"]       # 用 subtitle spec-like 在 ROI 内找字幕字形
SmoothMode = Literal["none", "bilateral"]
SpecEnableMode = Literal["always", "texture_only"]   # glare spec 可条件启用（不影响字幕）


@dataclass(frozen=True)
class PreprocessConfig:
    # ---------- subtitle ROI ----------
    subtitle_mask_mode: SubtitleMaskMode = "spec_roi"

    # 字幕高度 0.74
    subtitle_roi_y0_ratio: float = 0.74
    subtitle_roi_y1_ratio: float = 1.00

    # 通过连通域筛字幕：宽、矮、靠下（用于“分组后的块”）
    subtitle_min_area: int = 80
    subtitle_min_width_ratio: float = 0.12
    subtitle_max_height_ratio: float = 0.20
    subtitle_min_y_ratio: float = 0.74
    subtitle_center_x_min: float = 0.03
    subtitle_center_x_max: float = 0.97

    # 两阶段字幕：
    # (A) group：强膨胀仅用于“把一行字幕聚成块”做连通域筛选（不用于最终遮罩）
    subtitle_group_dilate_ksize: Tuple[int, int] = (25, 3)
    # (B) glyph：最终落地 invalid 只遮字形（轻补齐断笔，不抹掉字间隙）
    subtitle_glyph_dilate_ksize: Tuple[int, int] = (3, 1)

    # ---------- subtitle spec-like params (固定，不参与自适应) ----------
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
    log_edge_th: int = 35  # edge_density 统计阈值

# ---------- glare spec params (允许每视频自适应修改) ----------
    glare_spec_v_min: int = 230
    glare_spec_delta_sigma: float = 6.0
    glare_spec_delta_th: float = 24.0
    glare_spec_s_low: int = 55
    glare_spec_delta_strong_mul: float = 2.5

    # glare fill：解决“包边/没盖满”
    glare_fill_enable: bool = True
    glare_fill_v_offset: int = 12          # v_fill_min = glare_v_min - offset
    glare_fill_dilate_ksize: int = 11      # seed 膨胀半径（奇数）
    glare_fill_iters: int = 1

    # ---------- glare gating + CC + fallback ----------
    spec_enable_mode: SpecEnableMode = "texture_only"
    spec_texture_edge_density_min: float = 0.020

    glare_cc_min_area: int = 8
    glare_cc_max_area_ratio: float = 0.030     # 单块最大 <= 3% 全图像素
    glare_total_ratio_max: float = 0.080       # 总 spec > 8% -> 关闭 glare（认为炸了）
    glare_max_component_ratio_max: float = 0.050  # 最大块 > 5% -> 关闭 glare

    # 局部纹理过滤：防大块天空/平滑区误保留
    glare_cc_edge_density_min: float = 0.010


@dataclass
class PreprocessResult:
    intensity: np.ndarray     # uint8 HxW, smooth gray
    log: np.ndarray           # uint8 HxW
    valid_mask: np.ndarray    # uint8 HxW, 255 valid / 0 invalid
    spec_mask: np.ndarray     # uint8 HxW, 255 normal / 0 glare-like (soft suppression)
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


def _spec_like_tophat(
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
    spec_like = (V >= v_min) & (delta >= delta_th) & (S低 or delta很强)
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
    字幕屏蔽（两阶段，关键修复）：
    - ROI 中先用 sub_spec_like 得到“字形高亮”
    - (A) group：强膨胀把整行字幕连成一个块 -> 连通域筛“字幕块”
    - (B) glyph：最终 invalid 只落在字形像素（轻补齐断笔，不遮字间隙）
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
    glyph_u8 = (roi.astype(np.uint8) * 255)  # 原始字形，最终落地用它

    # (A) 分组强膨胀：仅用于让“整行字幕”连成块，便于连通域筛选
    gx, gy = cfg.subtitle_group_dilate_ksize
    gx, gy = max(1, int(gx)), max(1, int(gy))
    group_u8 = glyph_u8
    if gx > 1 or gy > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (gx, gy))
        group_u8 = cv2.dilate(glyph_u8, ker, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(group_u8, connectivity=8)
    dbg["components_total"] = int(max(0, num - 1))

    # 连通域筛“字幕块”
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
        # (B) 字形轻补齐：只补断笔，不抹字间隙
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


def _edge_density(log_u8: np.ndarray, valid_mask: np.ndarray, edge_th: int) -> float:
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
    glare spec_mask（修复“包边/没盖满”）：
    1) texture gating：非纹理场景可关闭（防天空/鸟整片红）
    2) seed→grow：从强高光边缘扩张，填充到光斑内部（解决包边）
    3) CC过滤：面积阈值 + 局部纹理阈值，防大块平滑误检
    4) fallback：总比例/最大块太大则整帧关闭 glare
    5) 字幕 invalid 区域强制 normal（双保险）
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

    # 1) gating
    ed = _edge_density(log_u8, valid_mask, cfg.log_edge_th)
    dbg["edge_density"] = float(ed)

    enabled = True
    if cfg.spec_enable_mode == "texture_only":
        enabled = (ed >= float(cfg.spec_texture_edge_density_min))
    dbg["enabled"] = bool(enabled)

    spec_mask = np.full((h, w), 255, dtype=np.uint8)
    if not enabled:
        return spec_mask, dbg

    # 2) seed：只在 valid
    seed_u8 = np.zeros((h, w), dtype=np.uint8)
    seed_u8[glare_spec_like & valid] = 255

    glare_u8 = seed_u8

    # 3) seed→grow fill：解决“包边/没盖满”
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

    # 4) CC过滤：面积 + 局部纹理
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

        # 局部纹理过滤：防大块平滑区（天空/纯色）误保留
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

    # 5) fallback（兜底关闭）
    valid_count = int(np.count_nonzero(valid))
    total_ratio = float(sum_area) / float(valid_count) if valid_count > 0 else 0.0
    max_ratio = float(max_comp_area) / float(valid_count) if valid_count > 0 else 0.0

    if total_ratio > float(cfg.glare_total_ratio_max):
        dbg["fallback"] = {"reason": "total_ratio_too_high", "total_ratio": total_ratio}
        return np.full((h, w), 255, dtype=np.uint8), dbg

    if max_ratio > float(cfg.glare_max_component_ratio_max):
        dbg["fallback"] = {"reason": "max_component_too_high", "max_ratio": max_ratio}
        return np.full((h, w), 255, dtype=np.uint8), dbg

    # 6) 输出 spec_mask（keep 为 glare-like）
    spec_mask[keep > 0] = 0

    # 字幕 invalid 区域强制 normal（双保险）
    spec_mask[~valid] = 255

    dbg["spec_ratio_valid_final"] = float(np.mean((spec_mask == 0)[valid])) if valid_count > 0 else 0.0
    return spec_mask, dbg


# ---------- 自适应：只用于 glare，不动 subtitle ----------
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
    输出：per-video cfg（只改 glare 参数） + debug 统计
    """
    if not stats_list:
        return base_cfg, {"used": False}

    v_p99 = float(np.median([s["v_p99"] for s in stats_list]))
    d_p995 = float(np.median([s["delta_p995"] for s in stats_list]))

    # v_min：靠近高亮尾部；过低会误伤鸟/天空，过高会漏
    new_v_min = int(np.clip(round(v_p99 - 5.0), 210, 245))

    # delta_th：根据片源整体对比度自适应（比例经验值）
    new_delta_th = float(np.clip(d_p995 * 0.45, 12.0, 60.0))

    # gating 阈值也可轻微自适应（可选，防某些片源永远 disabled）
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

    # 1) 字幕专用 spec（固定参数）
    sub_spec_like, dbg_sub_spec = _spec_like_tophat(
        bgr_u8,
        v_min=cfg.sub_spec_v_min,
        delta_sigma=cfg.sub_spec_delta_sigma,
        delta_th=cfg.sub_spec_delta_th,
        s_low=cfg.sub_spec_s_low,
        delta_strong_mul=cfg.sub_spec_delta_strong_mul,
    )

    # 2) valid_mask：字幕两阶段（强分组 + 字形落地）
    valid_mask, dbg_valid = compute_valid_mask_from_spec_roi(sub_spec_like, cfg)

    # 3) 眩光专用 spec（可自适应参数）
    glare_spec_like, dbg_glare_spec_base = _spec_like_tophat(
        bgr_u8,
        v_min=cfg.glare_spec_v_min,
        delta_sigma=cfg.glare_spec_delta_sigma,
        delta_th=cfg.glare_spec_delta_th,
        s_low=cfg.glare_spec_s_low,
        delta_strong_mul=cfg.glare_spec_delta_strong_mul,
    )

    # 4) glare spec_mask：gating + fill + CC + fallback + invalid 双保险
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


def apply_valid_mask_fill(gray_u8: np.ndarray, valid_mask: Optional[np.ndarray], sigma: float = 3.0) -> np.ndarray:
    """
    用 blur 填充 invalid 区域，避免 invalid=0 造成差分/匹配的硬边伪影。
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
