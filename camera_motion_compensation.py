# camera_motion_compensation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Literal
import numpy as np
import cv2

from preprocessing import ensure_gray_u8, apply_valid_mask_fill, combine_masks

RoiMode = Literal["corners", "strips", "corners+strips"]


@dataclass(frozen=True)
class CamMotionConfig:
    roi_mode: RoiMode = "strips"
    roi_frac: float = 0.35
    strip_frac: float = 0.18
    margin_frac: float = 0.02

    use_hanning: bool = True

    # Stage-1 (global PC) acceptance
    global_pc_resp_thresh: float = 0.30
    global_err_ratio_thresh: float = 0.92   # aligned_error / unaligned_error must be <= this
    global_min_improve: float = 0.5         # unaligned_error - aligned_error must be >= this

    # Stage-2 (ROI consensus PC)
    roi_pc_resp_thresh: float = 0.22
    st_max_corners: int = 300
    st_quality: float = 0.01
    st_min_distance: int = 7
    st_block_size: int = 7
    st_min_corners: int = 40

    mad_k: float = 3.0
    min_inlier_rois: int = 3

    # Stage-3 (ECC)
    enable_ecc_fallback: bool = True
    ecc_iterations: int = 60
    ecc_eps: float = 1e-4

    moving_thresh_px: float = 1.0
    max_abs_shift_px: float = 50.0

    save_roi_infos: bool = False


@dataclass
class CameraMotionResult:
    prev_aligned: np.ndarray
    T: Optional[np.ndarray]          # 2x3
    camera_moving: bool
    debug: Dict[str, Any]


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


def _shi_tomasi_count(roi_u8: np.ndarray, cfg: CamMotionConfig) -> int:
    corners = cv2.goodFeaturesToTrack(
        roi_u8,
        maxCorners=cfg.st_max_corners,
        qualityLevel=cfg.st_quality,
        minDistance=cfg.st_min_distance,
        blockSize=cfg.st_block_size,
        useHarrisDetector=False,
    )
    return 0 if corners is None else int(len(corners))


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
    """1D weighted median."""
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    c = np.cumsum(w)
    cutoff = 0.5 * float(c[-1])
    j = int(np.searchsorted(c, cutoff))
    return float(v[min(j, len(v) - 1)])


def _roi_consensus_shift(prev_u8: np.ndarray, curr_u8: np.ndarray, cfg: CamMotionConfig) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    H, W = curr_u8.shape[:2]
    rois = _make_rois(H, W, cfg)

    meas: List[Dict[str, Any]] = []
    for (x, y, w, h, name) in rois:
        pr = prev_u8[y:y+h, x:x+w]
        cr = curr_u8[y:y+h, x:x+w]

        n_corners = _shi_tomasi_count(pr, cfg)
        (dx, dy), resp = _phase_corr_shift(pr, cr, cfg.use_hanning)

        corner_ratio = min(1.0, n_corners / max(1.0, float(cfg.st_min_corners)))
        weight = max(0.0, resp * corner_ratio)

        info = {
            "roi": name,
            "rect": [int(x), int(y), int(w), int(h)],
            "dx": float(dx),
            "dy": float(dy),
            "resp": float(resp),
            "corners": int(n_corners),
            "weight": float(weight),
        }
        meas.append(info)

    # filter usable
    usable = [
        m for m in meas
        if (m["resp"] >= cfg.roi_pc_resp_thresh) and (m["corners"] >= cfg.st_min_corners) and (m["weight"] > 0)
    ]

    dbg: Dict[str, Any] = {
        "roi_total": int(len(meas)),
        "roi_usable": int(len(usable)),
        "roi_pc_resp_thresh": float(cfg.roi_pc_resp_thresh),
        "st_min_corners": int(cfg.st_min_corners),
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

    # MAD in 2D residual
    res = np.sqrt((dxs - dx0) ** 2 + (dys - dy0) ** 2)
    mad = float(np.median(np.abs(res - np.median(res))) + 1e-6)
    thr = cfg.mad_k * (1.4826 * mad + 1e-6)

    inlier = res <= max(thr, 1.0)  # allow at least 1px tolerance
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

        # mask: 0/255 uint8 accepted
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
    Estimate translation from prev -> curr using LoG features (recommended).
    - valid_mask: subtitle/invalid region (same each frame usually)
    - prev_spec_mask/curr_spec_mask: per-frame spec mask (255 normal, 0 specular)
    - warp_src: if provided, warp this image instead of prev_feat (e.g., warp intensity with LoG-estimated T)
    """
    prev_feat = ensure_gray_u8(prev_feat)
    curr_feat = ensure_gray_u8(curr_feat)

    # combine masks: pixel is valid only if it's valid AND not specular in both frames
    comb_mask = combine_masks(valid_mask, prev_spec_mask, curr_spec_mask)

    prev_use = apply_valid_mask_fill(prev_feat, comb_mask)
    curr_use = apply_valid_mask_fill(curr_feat, comb_mask)

    H, W = curr_use.shape[:2]
    debug: Dict[str, Any] = {"method": "identity"}

    # ---------- Stage 1: global phase correlation ----------
    (dx, dy), resp = _phase_corr_shift(prev_use, curr_use, cfg.use_hanning)

    # reliability via "alignment reduces median abs error"
    err0 = _median_abs_error(curr_use, prev_use, comb_mask)
    prev_warp_test = _warp_u8(prev_use, dx, dy)
    err1 = _median_abs_error(curr_use, prev_warp_test, comb_mask)

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
        shift2, roi_dbg = _roi_consensus_shift(prev_use, curr_use, cfg)
        debug["roi_consensus"] = roi_dbg

        if shift2 is not None:
            dx2, dy2 = shift2
            dx2 = float(np.clip(dx2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            dy2 = float(np.clip(dy2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))

            prev_warp_test2 = _warp_u8(prev_use, dx2, dy2)
            err2 = _median_abs_error(curr_use, prev_warp_test2, comb_mask)

            debug["roi_consensus_check"] = {
                "dx": float(dx2), "dy": float(dy2),
                "err_unaligned": float(err0),
                "err_aligned": float(err2),
                "err_ratio": float(err2 / (err0 + 1e-6)),
                "improve": float(err0 - err2),
            }

            # accept if also improves
            if (err2 / (err0 + 1e-6) <= cfg.global_err_ratio_thresh) and (err0 - err2 >= cfg.global_min_improve):
                used_dxdy = (dx2, dy2)
                debug["method"] = "roi_consensus_phase_correlation"

    # ---------- Stage 3: ECC fallback ----------
    if used_dxdy is None and cfg.enable_ecc_fallback:
        ecc_shift, ecc_dbg = _ecc_translation(prev_use, curr_use, comb_mask, cfg)
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
