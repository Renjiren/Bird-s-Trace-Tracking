# camera_motion_compensation.py
# step2 输出prev_aligned、T(2x3)、camera_moving
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Literal
import numpy as np
import cv2

from preprocessing import ensure_gray_u8, preprocess_frame, PreprocessConfig

RoiMode = Literal["corners", "strips", "corners+strips"]


@dataclass(frozen=True)
class CamMotionConfig:
    # ---------- ROI selection ----------
    roi_mode: RoiMode = "strips"
    roi_frac: float = 0.35
    strip_frac: float = 0.18
    margin_frac: float = 0.02
    use_hanning: bool = True

    # ROI 有效像素比例门槛（避免字幕 hard-mask 导致 BOT strip 失效）
    roi_valid_frac_min: float = 0.45

    # ---------- Stage-1 (global PC) acceptance ----------
    global_pc_resp_thresh: float = 0.30
    global_err_ratio_thresh: float = 0.92
    global_min_improve: float = 0.5

    # ---------- Stage-2 (ROI consensus PC) ----------
    roi_pc_resp_thresh: float = 0.22
    st_max_corners: int = 300
    st_quality: float = 0.01
    st_min_distance: int = 7
    st_block_size: int = 7
    st_min_corners: int = 40

    mad_k: float = 3.0
    min_inlier_rois: int = 3

    # ---------- Stage-3 (ECC) ----------
    enable_ecc_fallback: bool = True
    ecc_iterations: int = 60
    ecc_eps: float = 1e-4

    # ---------- moving decision ----------
    moving_thresh_px: float = 1.0
    max_abs_shift_px: float = 50.0

    # ---------- soft spec suppression ----------
    # spec_mask: 255 normal, 0 glare-like / spec-like
    # 注意：这里是 soft，不做硬剔除
    use_soft_spec: bool = True

    # soft blend：out = w*img + (1-w)*blur(img)
    # normal: w=1; spec: w=soft_spec_alpha（0=完全替换成blur；0.3=保留一些）
    soft_spec_alpha: float = 0.0
    soft_spec_blur_sigma: float = 3.0
    soft_spec_dilate_ksize: int = 5

    # 如果 spec_union 在 valid 区域占比太大，说明 spec_mask 可能“炸了/误检大片天空”
    # 这种情况下宁愿不用 soft_spec（避免把整帧模糊掉）
    soft_spec_max_ratio: float = 0.25

    # ---------- error mask strategy ----------
    # 误差评估时可选排除 spec_union（只影响 acceptance，不影响 shift 估计）
    err_mask_exclude_spec: bool = True
    err_mask_spec_max_ratio: float = 0.15  # spec_union <= 15% 才排除，否则不用排除
    err_mask_min_valid_frac: float = 0.20  # 排除后至少保留这么多 valid 才启用

    save_roi_infos: bool = False


@dataclass
class CameraMotionResult:
    prev_aligned: np.ndarray
    T: Optional[np.ndarray]          # 2x3
    camera_moving: bool
    debug: Dict[str, Any]


# ----------------- helpers -----------------
def _make_rois(H: int, W: int, cfg: CamMotionConfig) -> List[Tuple[int, int, int, int, str]]:
    mx = int(W * cfg.margin_frac)
    my = int(H * cfg.margin_frac)

    def clamp_rect(x, y, w, h):
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        return x, y, w, h

    rois: List[Tuple[int, int, int, int, str]] = []

    if cfg.roi_mode in ("corners", "corners+strips"):
        roi_w = max(32, int(W * cfg.roi_frac))
        roi_h = max(32, int(H * cfg.roi_frac))
        roi_w = min(roi_w, W - 2 * mx)
        roi_h = min(roi_h, H - 2 * my)

        rois.append((*clamp_rect(mx, my, roi_w, roi_h), "TL"))
        rois.append((*clamp_rect(W - mx - roi_w, my, roi_w, roi_h), "TR"))
        rois.append((*clamp_rect(mx, H - my - roi_h, roi_w, roi_h), "BL"))
        rois.append((*clamp_rect(W - mx - roi_w, H - my - roi_h, roi_w, roi_h), "BR"))

    if cfg.roi_mode in ("strips", "corners+strips"):
        t = max(24, int(min(H, W) * cfg.strip_frac))
        rois.append((*clamp_rect(mx, my, W - 2 * mx, t), "TOP"))
        rois.append((*clamp_rect(mx, H - my - t, W - 2 * mx, t), "BOT"))
        rois.append((*clamp_rect(mx, my, t, H - 2 * my), "LFT"))
        rois.append((*clamp_rect(W - mx - t, my, t, H - 2 * my), "RGT"))

    return rois


def _phase_corr_shift(a_u8: np.ndarray, b_u8: np.ndarray, use_hanning: bool) -> Tuple[Tuple[float, float], float]:
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    if use_hanning:
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        shift, resp = cv2.phaseCorrelate(a, b, win)
    else:
        shift, resp = cv2.phaseCorrelate(a, b)
    return (float(shift[0]), float(shift[1])), float(resp)


def _warp_u8(img_u8: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = img_u8.shape[:2]
    T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img_u8, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _median_abs_error(a_u8: np.ndarray, b_u8: np.ndarray, mask_u8: Optional[np.ndarray]) -> float:
    diff = cv2.absdiff(a_u8, b_u8)
    if mask_u8 is not None and np.any(mask_u8 > 0):
        vals = diff[mask_u8 > 0]
    else:
        vals = diff.reshape(-1)
    if vals.size == 0:
        return float(np.median(diff))
    return float(np.median(vals))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    c = np.cumsum(w)
    cutoff = 0.5 * float(c[-1])
    j = int(np.searchsorted(c, cutoff))
    return float(v[min(j, len(v) - 1)])


def _hard_fill_invalid(img_u8: np.ndarray, hard_mask_u8: Optional[np.ndarray], sigma: float = 3.0) -> np.ndarray:
    """
    hard mask（字幕等无效区）处理：不让相位相关看到“黑洞/蓝洞”。
    用 blur 图替换 invalid 区域，保证连续性。
    """
    if hard_mask_u8 is None:
        return img_u8
    if hard_mask_u8.dtype != np.uint8:
        hard_mask_u8 = hard_mask_u8.astype(np.uint8)
    if not np.any(hard_mask_u8 == 0):
        return img_u8

    img = img_u8
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(max(0.0, sigma)), sigmaY=float(max(0.0, sigma)))
    out = img.copy()
    out[hard_mask_u8 == 0] = blur[hard_mask_u8 == 0]
    return out


def _make_spec_union_bad(prev_spec_mask: Optional[np.ndarray],
                         curr_spec_mask: Optional[np.ndarray],
                         hard_mask_u8: Optional[np.ndarray],
                         cfg: CamMotionConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    spec_union_bad: 255 表示 “在任一帧被标为 spec”，用于 soft suppression。
    """
    dbg: Dict[str, Any] = {"used": False, "ratio_valid": 0.0, "dilated": False}
    if (prev_spec_mask is None) and (curr_spec_mask is None):
        return np.zeros((1, 1), dtype=np.uint8), dbg  # caller会忽略 shape 不匹配时重新建
    # build bad map
    if prev_spec_mask is None:
        bad = (curr_spec_mask == 0)
    elif curr_spec_mask is None:
        bad = (prev_spec_mask == 0)
    else:
        bad = (prev_spec_mask == 0) | (curr_spec_mask == 0)

    bad_u8 = (bad.astype(np.uint8) * 255)

    # optional dilate to cover fringes
    k = int(max(1, cfg.soft_spec_dilate_ksize))
    if k % 2 == 0:
        k += 1
    if k >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bad_u8 = cv2.dilate(bad_u8, ker, iterations=1)
        dbg["dilated"] = True

    # ratio on valid
    if hard_mask_u8 is not None and hard_mask_u8.shape == bad_u8.shape:
        valid = (hard_mask_u8 > 0)
        valid_n = int(np.count_nonzero(valid))
        if valid_n > 0:
            dbg["ratio_valid"] = float(np.mean((bad_u8 > 0)[valid]))
    else:
        dbg["ratio_valid"] = float(np.mean(bad_u8 > 0))

    dbg["used"] = True
    return bad_u8, dbg


def _apply_soft_spec_suppression(img_u8: np.ndarray,
                                 spec_bad_u8: Optional[np.ndarray],
                                 hard_mask_u8: Optional[np.ndarray],
                                 cfg: CamMotionConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    soft spec：对 spec_union 区域做 blur 替换/混合，降低其影响，但不硬剔除。
    """
    dbg: Dict[str, Any] = {"enabled": False, "bypass_reason": None}

    if (not cfg.use_soft_spec) or (spec_bad_u8 is None):
        dbg["bypass_reason"] = "disabled_or_none"
        return img_u8, dbg

    if spec_bad_u8.shape != img_u8.shape:
        dbg["bypass_reason"] = "shape_mismatch"
        return img_u8, dbg

    if not np.any(spec_bad_u8 > 0):
        dbg["bypass_reason"] = "no_spec_pixels"
        return img_u8, dbg

    # ratio gating
    if hard_mask_u8 is not None and hard_mask_u8.shape == img_u8.shape:
        valid = (hard_mask_u8 > 0)
        valid_n = int(np.count_nonzero(valid))
        ratio = float(np.mean((spec_bad_u8 > 0)[valid])) if valid_n > 0 else 0.0
    else:
        ratio = float(np.mean(spec_bad_u8 > 0))

    dbg["spec_ratio_valid"] = ratio
    if ratio > float(cfg.soft_spec_max_ratio):
        dbg["bypass_reason"] = "spec_ratio_too_high"
        return img_u8, dbg

    sigma = float(max(0.0, cfg.soft_spec_blur_sigma))
    blur = cv2.GaussianBlur(img_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)

    alpha = float(np.clip(cfg.soft_spec_alpha, 0.0, 1.0))
    out = img_u8.copy()

    # only apply on valid area (可选)
    if hard_mask_u8 is not None and hard_mask_u8.shape == img_u8.shape:
        region = (spec_bad_u8 > 0) & (hard_mask_u8 > 0)
    else:
        region = (spec_bad_u8 > 0)

    if alpha <= 1e-6:
        out[region] = blur[region]
    else:
        # float blend
        out_f = out.astype(np.float32)
        blur_f = blur.astype(np.float32)
        out_f[region] = alpha * out_f[region] + (1.0 - alpha) * blur_f[region]
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    dbg["enabled"] = True
    dbg["alpha"] = alpha
    dbg["blur_sigma"] = sigma
    return out, dbg


def _shi_tomasi_count(roi_u8: np.ndarray, roi_mask_u8: Optional[np.ndarray], cfg: CamMotionConfig) -> int:
    corners = cv2.goodFeaturesToTrack(
        roi_u8,
        maxCorners=cfg.st_max_corners,
        qualityLevel=cfg.st_quality,
        minDistance=cfg.st_min_distance,
        blockSize=cfg.st_block_size,
        useHarrisDetector=False,
        mask=roi_mask_u8
    )
    return 0 if corners is None else int(len(corners))


def _roi_consensus_shift(prev_u8: np.ndarray,
                         curr_u8: np.ndarray,
                         roi_mask_u8: Optional[np.ndarray],
                         cfg: CamMotionConfig) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    H, W = curr_u8.shape[:2]
    rois = _make_rois(H, W, cfg)

    meas: List[Dict[str, Any]] = []
    for (x, y, w, h, name) in rois:
        pr = prev_u8[y:y+h, x:x+w]
        cr = curr_u8[y:y+h, x:x+w]

        if roi_mask_u8 is not None and roi_mask_u8.shape == prev_u8.shape:
            m = roi_mask_u8[y:y+h, x:x+w]
            valid_frac = float(np.mean(m > 0))
            if valid_frac < float(cfg.roi_valid_frac_min):
                meas.append({
                    "roi": name, "rect": [int(x), int(y), int(w), int(h)],
                    "skip": True, "reason": "low_valid_frac", "valid_frac": valid_frac
                })
                continue
            m_use = m
        else:
            valid_frac = 1.0
            m_use = None

        n_corners = _shi_tomasi_count(pr, m_use, cfg)
        (dx, dy), resp = _phase_corr_shift(pr, cr, cfg.use_hanning)

        corner_ratio = min(1.0, n_corners / max(1.0, float(cfg.st_min_corners)))
        # 加一个 valid_frac 的权重，让 ROI “有效内容多”的更可信
        weight = max(0.0, resp * corner_ratio * np.sqrt(valid_frac))

        info = {
            "roi": name,
            "rect": [int(x), int(y), int(w), int(h)],
            "dx": float(dx),
            "dy": float(dy),
            "resp": float(resp),
            "corners": int(n_corners),
            "valid_frac": float(valid_frac),
            "weight": float(weight),
            "skip": False,
        }
        meas.append(info)

    usable = [
        m for m in meas
        if (not m.get("skip", False))
        and (m["resp"] >= cfg.roi_pc_resp_thresh)
        and (m["corners"] >= cfg.st_min_corners)
        and (m["weight"] > 0)
    ]

    dbg: Dict[str, Any] = {
        "roi_total": int(len(meas)),
        "roi_usable": int(len(usable)),
        "roi_pc_resp_thresh": float(cfg.roi_pc_resp_thresh),
        "st_min_corners": int(cfg.st_min_corners),
        "roi_valid_frac_min": float(cfg.roi_valid_frac_min),
    }
    if cfg.save_roi_infos:
        dbg["roi_infos"] = meas

    if len(usable) < cfg.min_inlier_rois:
        dbg["reason"] = "not_enough_usable_rois"
        return None, dbg

    dxs = np.array([m["dx"] for m in usable], dtype=np.float32)
    dys = np.array([m["dy"] for m in usable], dtype=np.float32)
    ws = np.array([m["weight"] for m in usable], dtype=np.float32)

    dx0 = _weighted_median(dxs, ws)
    dy0 = _weighted_median(dys, ws)

    res = np.sqrt((dxs - dx0) ** 2 + (dys - dy0) ** 2)
    mad = float(np.median(np.abs(res - np.median(res))) + 1e-6)
    thr = cfg.mad_k * (1.4826 * mad + 1e-6)

    inlier = res <= max(thr, 1.0)
    inliers = [u for u, keep in zip(usable, inlier.tolist()) if keep]

    dbg["dx0"] = float(dx0)
    dbg["dy0"] = float(dy0)
    dbg["mad"] = float(mad)
    dbg["inliers"] = int(len(inliers))
    dbg["inlier_thr"] = float(max(thr, 1.0))

    if len(inliers) < cfg.min_inlier_rois:
        dbg["reason"] = "not_enough_inliers_after_mad"
        return None, dbg

    dxs2 = np.array([m["dx"] for m in inliers], dtype=np.float32)
    dys2 = np.array([m["dy"] for m in inliers], dtype=np.float32)
    ws2 = np.array([m["weight"] for m in inliers], dtype=np.float32)
    dx = _weighted_median(dxs2, ws2)
    dy = _weighted_median(dys2, ws2)

    dbg["dx_final"] = float(dx)
    dbg["dy_final"] = float(dy)
    dbg["resp_mean_inliers"] = float(np.mean([m["resp"] for m in inliers]))

    return (float(dx), float(dy)), dbg


def _ecc_translation(prev_u8: np.ndarray, curr_u8: np.ndarray, mask_u8: Optional[np.ndarray], cfg: CamMotionConfig) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"ecc_enabled": True}
    try:
        template = curr_u8.astype(np.float32) / 255.0
        inp = prev_u8.astype(np.float32) / 255.0

        warp = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(cfg.ecc_iterations),
                    float(cfg.ecc_eps))

        cc, warp = cv2.findTransformECC(
            templateImage=template,
            inputImage=inp,
            warpMatrix=warp,
            motionType=cv2.MOTION_TRANSLATION,
            criteria=criteria,
            inputMask=mask_u8
        )

        dx = float(warp[0, 2])
        dy = float(warp[1, 2])
        dbg["ecc_cc"] = float(cc)
        dbg["dx"] = dx
        dbg["dy"] = dy
        return (dx, dy), dbg
    except cv2.error as e:
        dbg["reason"] = "ecc_failed"
        dbg["cv2_error"] = str(e)[:200]
        return None, dbg


# ----------------- main estimation -----------------
def estimate_camera_translation(
    prev_feat: np.ndarray,
    curr_feat: np.ndarray,
    valid_mask: Optional[np.ndarray],
    prev_spec_mask: Optional[np.ndarray],
    curr_spec_mask: Optional[np.ndarray],
    cfg: CamMotionConfig,
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    估计 prev -> curr 的平移（推荐输入 LoG 特征）。
    - valid_mask: hard mask（字幕/无效区），255 valid / 0 invalid
    - spec_mask: soft mask（眩光/强亮/鸟体误判等），255 normal / 0 spec-like；只做 soft suppression
    """
    prev_feat = ensure_gray_u8(prev_feat)
    curr_feat = ensure_gray_u8(curr_feat)

    H, W = curr_feat.shape[:2]
    if valid_mask is None:
        hard_mask = np.full((H, W), 255, dtype=np.uint8)
    else:
        hard_mask = (valid_mask.astype(np.uint8) if valid_mask.dtype == np.uint8 else valid_mask.astype(np.uint8))

    # 1) hard invalid fill：避免相位相关被“空洞”扰动
    prev_use = _hard_fill_invalid(prev_feat, hard_mask, sigma=3.0)
    curr_use = _hard_fill_invalid(curr_feat, hard_mask, sigma=3.0)

    debug: Dict[str, Any] = {"method": "identity"}
    debug["hard_mask"] = {
        "valid_ratio": float(np.mean(hard_mask > 0)),
        "invalid_ratio": float(np.mean(hard_mask == 0)),
    }

    # 2) spec_union -> soft suppression（不硬剔除）
    spec_dbg = {"used": False}
    spec_bad_u8: Optional[np.ndarray] = None
    if cfg.use_soft_spec and (prev_spec_mask is not None or curr_spec_mask is not None):
        spec_bad_u8, spec_union_dbg = _make_spec_union_bad(prev_spec_mask, curr_spec_mask, hard_mask, cfg)
        # 修正 shape（上面 _make_spec_union_bad 在极端输入可能返回 1x1）
        if spec_bad_u8.shape != (H, W):
            # 兜底：直接按当前 shape 重算
            if prev_spec_mask is None:
                bad = (curr_spec_mask == 0)
            elif curr_spec_mask is None:
                bad = (prev_spec_mask == 0)
            else:
                bad = (prev_spec_mask == 0) | (curr_spec_mask == 0)
            spec_bad_u8 = (bad.astype(np.uint8) * 255)
        prev_use, sdbg1 = _apply_soft_spec_suppression(prev_use, spec_bad_u8, hard_mask, cfg)
        curr_use, sdbg2 = _apply_soft_spec_suppression(curr_use, spec_bad_u8, hard_mask, cfg)
        spec_dbg = {"union": spec_union_dbg, "prev": sdbg1, "curr": sdbg2}

    debug["soft_spec"] = spec_dbg

    # 3) error mask（用于 acceptance / ECC）：以 hard 为主，必要时排除 spec_union
    err_mask = hard_mask.copy()
    if cfg.err_mask_exclude_spec and (spec_bad_u8 is not None) and (spec_bad_u8.shape == err_mask.shape):
        valid = (hard_mask > 0)
        valid_n = int(np.count_nonzero(valid))
        if valid_n > 0:
            spec_ratio = float(np.mean((spec_bad_u8 > 0)[valid]))
            if spec_ratio <= float(cfg.err_mask_spec_max_ratio):
                tmp = err_mask.copy()
                tmp[spec_bad_u8 > 0] = 0
                keep_frac = float(np.mean(tmp > 0))
                if keep_frac >= float(cfg.err_mask_min_valid_frac):
                    err_mask = tmp
                    debug["err_mask"] = {"exclude_spec": True, "spec_ratio": spec_ratio, "keep_frac": keep_frac}
                else:
                    debug["err_mask"] = {"exclude_spec": False, "reason": "keep_frac_too_low", "keep_frac": keep_frac}
            else:
                debug["err_mask"] = {"exclude_spec": False, "reason": "spec_ratio_too_high", "spec_ratio": spec_ratio}
        else:
            debug["err_mask"] = {"exclude_spec": False, "reason": "no_valid_pixels"}
    else:
        debug["err_mask"] = {"exclude_spec": False, "reason": "disabled_or_no_spec"}

    # ---------- Stage 1: global phase correlation ----------
    (dx, dy), resp = _phase_corr_shift(prev_use, curr_use, cfg.use_hanning)

    err0 = _median_abs_error(curr_use, prev_use, err_mask)
    prev_warp_test = _warp_u8(prev_use, dx, dy)
    err1 = _median_abs_error(curr_use, prev_warp_test, err_mask)

    debug["global_pc"] = {
        "dx": float(dx), "dy": float(dy), "resp": float(resp),
        "err_unaligned": float(err0),
        "err_aligned": float(err1),
        "err_ratio": float(err1 / (err0 + 1e-6)),
        "improve": float(err0 - err1),
        "resp_thresh": float(cfg.global_pc_resp_thresh),
        "err_ratio_thresh": float(cfg.global_err_ratio_thresh),
        "min_improve": float(cfg.global_min_improve),
    }

    global_ok = (
        resp >= cfg.global_pc_resp_thresh and
        abs(dx) <= cfg.max_abs_shift_px and abs(dy) <= cfg.max_abs_shift_px and
        (err1 / (err0 + 1e-6) <= cfg.global_err_ratio_thresh) and
        (err0 - err1 >= cfg.global_min_improve)
    )

    used_dxdy: Optional[Tuple[float, float]] = None
    if global_ok:
        used_dxdy = (float(dx), float(dy))
        debug["method"] = "global_phase_correlation"

    # ---------- Stage 2: ROI consensus + MAD ----------
    if used_dxdy is None:
        # ROI 内角点与 valid_frac 使用 err_mask（更“稳定”）
        shift2, roi_dbg = _roi_consensus_shift(prev_use, curr_use, err_mask, cfg)
        debug["roi_consensus"] = roi_dbg

        if shift2 is not None:
            dx2, dy2 = shift2
            dx2 = float(np.clip(dx2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            dy2 = float(np.clip(dy2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))

            prev_warp_test2 = _warp_u8(prev_use, dx2, dy2)
            err2 = _median_abs_error(curr_use, prev_warp_test2, err_mask)

            debug["roi_consensus_check"] = {
                "dx": float(dx2), "dy": float(dy2),
                "err_unaligned": float(err0),
                "err_aligned": float(err2),
                "err_ratio": float(err2 / (err0 + 1e-6)),
                "improve": float(err0 - err2),
            }

            if (err2 / (err0 + 1e-6) <= cfg.global_err_ratio_thresh) and (err0 - err2 >= cfg.global_min_improve):
                used_dxdy = (dx2, dy2)
                debug["method"] = "roi_consensus_phase_correlation"

    # ---------- Stage 3: ECC fallback ----------
    if used_dxdy is None and cfg.enable_ecc_fallback:
        ecc_shift, ecc_dbg = _ecc_translation(prev_use, curr_use, err_mask, cfg)
        debug["ecc"] = ecc_dbg
        if ecc_shift is not None:
            dx3, dy3 = ecc_shift
            dx3 = float(np.clip(dx3, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            dy3 = float(np.clip(dy3, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            used_dxdy = (dx3, dy3)
            debug["method"] = "ecc_translation"

    # final warp
    if used_dxdy is None:
        out = warp_src if warp_src is not None else prev_feat
        out = ensure_gray_u8(out)
        return CameraMotionResult(prev_aligned=out.copy(), T=None, camera_moving=False, debug=debug)

    dx_f, dy_f = used_dxdy
    T = np.array([[1.0, 0.0, dx_f], [0.0, 1.0, dy_f]], dtype=np.float32)

    src = warp_src if warp_src is not None else prev_feat
    src = ensure_gray_u8(src)
    prev_aligned = cv2.warpAffine(src, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    shift_norm = float(np.hypot(dx_f, dy_f))
    camera_moving = bool(shift_norm >= float(cfg.moving_thresh_px))

    debug.update({
        "final_dx": float(dx_f),
        "final_dy": float(dy_f),
        "shift_norm": float(shift_norm),
        "camera_moving": camera_moving,
    })

    return CameraMotionResult(prev_aligned=prev_aligned, T=T, camera_moving=camera_moving, debug=debug)


# ----------------- convenience: run step2 without saving step1 -----------------
def estimate_camera_translation_from_bgr(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    feature: Literal["log", "intensity"] = "log",
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    只跑 step2 的推荐入口：直接给两帧 bgr，内部实时跑 step1（不落盘、不保存）。
    - feature="log"：用 LoG 估计相机平移（推荐）
    - warp_src：如果你想“用 LoG 得到的 T 去 warp intensity”，这里传 prev_pre.intensity 即可
    """
    pre_prev = preprocess_frame(prev_bgr, pre_cfg)
    pre_curr = preprocess_frame(curr_bgr, pre_cfg)

    if feature == "log":
        prev_feat = pre_prev.log
        curr_feat = pre_curr.log
    else:
        prev_feat = pre_prev.intensity
        curr_feat = pre_curr.intensity

    # 默认 warp_src：如果你没传，就 warp prev_feat
    src = warp_src
    if src is None:
        src = prev_feat

    return estimate_camera_translation(
        prev_feat=prev_feat,
        curr_feat=curr_feat,
        valid_mask=pre_prev.valid_mask,       # hard subtitle mask
        prev_spec_mask=pre_prev.spec_mask,    # soft
        curr_spec_mask=pre_curr.spec_mask,    # soft
        cfg=cam_cfg,
        warp_src=src,
    )

