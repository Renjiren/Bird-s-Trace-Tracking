# camera_motion_compensation.py
# Step2: output prev_aligned、T(2x3)、camera_moving

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Literal
import numpy as np
import cv2

from preprocessing import ensure_gray_u8, preprocess_frame, PreprocessConfig, apply_valid_mask_fill

RoiMode = Literal["corners", "strips", "corners+strips"]
EccMode = Literal["euclidean", "affine"]


@dataclass(frozen=True)
class CamMotionConfig:
    # ---------- ROI selection ----------
    roi_mode: RoiMode = "strips"
    roi_frac: float = 0.35
    strip_frac: float = 0.18
    margin_frac: float = 0.02
    use_hanning: bool = True
    roi_valid_frac_min: float = 0.45

    # ---------- Stage-1/2 acceptance ----------
    global_pc_resp_thresh: float = 0.30
    global_err_ratio_thresh: float = 0.92
    global_min_improve: float = 0.5

    roi_pc_resp_thresh: float = 0.22
    st_max_corners: int = 300
    st_quality: float = 0.01
    st_min_distance: int = 7
    st_block_size: int = 7
    st_min_corners: int = 40

    mad_k: float = 3.0
    min_inlier_rois: int = 3

    # ---------- Stage-3 (ECC fallback) ----------
    enable_ecc_fallback: bool = True
    ecc_order: Tuple[EccMode, ...] = ("euclidean", "affine")  # Euclidean -> Affine(=Affine2)
    ecc_iterations: int = 70
    ecc_eps: float = 1e-4

    # ---------- moving decision ----------
    moving_thresh_px: float = 1.0
    max_abs_shift_px: float = 60.0

    # ---------- spec_mask weighting suppression ----------
    # spec_mask: 255 normal, 0 glare-like
    use_spec_weight: bool = True
    spec_weight: float = 0.25               # spec region weight (the smaller the value, the more suppressed)
    spec_dilate_ksize: int = 5              # covering glare margin
    spec_weight_max_ratio: float = 0.25     # If the spec component is too large, do not use spec weights at all (to avoid false positives on large areas of the sky).

    save_roi_infos: bool = False


@dataclass
class CameraMotionResult:
    prev_aligned: np.ndarray
    T: Optional[np.ndarray]          # 2x3
    camera_moving: bool
    debug: Dict[str, Any]


def odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k + (k % 2 == 0)


def make_rois(H: int, W: int, cfg: CamMotionConfig) -> List[Tuple[int, int, int, int, str]]:
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


def warp_u8(img_u8: np.ndarray, T_2x3: np.ndarray) -> np.ndarray:
    H, W = img_u8.shape[:2]
    return cv2.warpAffine(img_u8, T_2x3.astype(np.float32), (W, H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def median_abs_error(a_u8: np.ndarray, b_u8: np.ndarray, mask_u8: Optional[np.ndarray]) -> float:
    diff = cv2.absdiff(a_u8, b_u8)
    if mask_u8 is not None and np.any(mask_u8 > 0):
        vals = diff[mask_u8 > 0]
    else:
        vals = diff.reshape(-1)
    if vals.size == 0:
        return float(np.median(diff))
    return float(np.median(vals))


def weighted_zero_mean(img_u8: np.ndarray, w_f32: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32)
    w = w_f32.astype(np.float32)
    s = float(np.sum(w))
    if s <= 1e-6:
        return x
    mu = float(np.sum(x * w) / s)
    return (x - mu) * w


def phase_corr_shift(a_u8: np.ndarray, b_u8: np.ndarray, use_hanning: bool) -> Tuple[Tuple[float, float], float]:
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    if use_hanning:
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        shift, resp = cv2.phaseCorrelate(a, b, win)
    else:
        shift, resp = cv2.phaseCorrelate(a, b)
    return (float(shift[0]), float(shift[1])), float(resp)


def shi_tomasi_count(roi_u8: np.ndarray, roi_mask_u8: Optional[np.ndarray], cfg: CamMotionConfig) -> int:
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


def build_weight_and_err_mask(
    valid_mask: Optional[np.ndarray],
    prev_spec_mask: Optional[np.ndarray],
    curr_spec_mask: Optional[np.ndarray],
    cfg: CamMotionConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    统一生成：
      - w_f32: phase-corr 输入的软权重（valid=1, invalid=0, spec=spec_weight）
      - err_mask_u8: 误差评估/ECC 的硬 mask（默认 valid；可在 spec 比例小的时候排除 spec）
    """
    dbg: Dict[str, Any] = {}

    if valid_mask is None:
        raise ValueError("valid_mask is required for stable motion estimation.")
    hard = valid_mask.astype(np.uint8) if valid_mask.dtype != np.uint8 else valid_mask
    valid = (hard > 0)

    # spec union
    spec_union = None
    if prev_spec_mask is not None or curr_spec_mask is not None:
        if prev_spec_mask is None:
            spec_union = (curr_spec_mask == 0)
        elif curr_spec_mask is None:
            spec_union = (prev_spec_mask == 0)
        else:
            spec_union = (prev_spec_mask == 0) | (curr_spec_mask == 0)

    spec_ratio = 0.0
    if spec_union is not None and np.any(valid):
        spec_ratio = float(np.mean(spec_union[valid]))

    dbg["valid_ratio"] = float(np.mean(valid))
    dbg["spec_ratio_valid"] = float(spec_ratio)

    # weight map
    w = valid.astype(np.float32)
    use_spec = bool(cfg.use_spec_weight and spec_union is not None and spec_ratio <= float(cfg.spec_weight_max_ratio))
    dbg["use_spec_weight"] = use_spec
    dbg["spec_weight"] = float(cfg.spec_weight)

    if use_spec:
        bad = spec_union.astype(np.uint8) * 255
        k = odd(cfg.spec_dilate_ksize)
        if k >= 3:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bad = cv2.dilate(bad, ker, iterations=1)
        bad_bool = (bad > 0) & valid
        w[bad_bool] *= float(np.clip(cfg.spec_weight, 0.0, 1.0))

    # err mask：硬 mask，默认 valid；如果 spec 占比不大，则 ECC/误差评估时排除 spec（更稳）
    err = hard.copy()
    if spec_union is not None and spec_ratio <= float(cfg.spec_weight_max_ratio):
        err[(spec_union & valid)] = 0

    keep_frac = float(np.mean(err > 0))
    if keep_frac < 0.20:
        # 保底：别把 mask 削太狠
        err = hard.copy()
        dbg["err_mask_fallback"] = "keep_frac_too_low"
    dbg["err_keep_frac"] = float(np.mean(err > 0))

    return w, err, dbg


def roi_consensus_shift(prev_u8: np.ndarray,
                         curr_u8: np.ndarray,
                         mask_u8: np.ndarray,
                         cfg: CamMotionConfig) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    H, W = curr_u8.shape[:2]
    rois = make_rois(H, W, cfg)

    meas: List[Dict[str, Any]] = []
    for (x, y, w, h, name) in rois:
        pr = prev_u8[y:y+h, x:x+w]
        cr = curr_u8[y:y+h, x:x+w]
        m = mask_u8[y:y+h, x:x+w]

        valid_frac = float(np.mean(m > 0))
        if valid_frac < float(cfg.roi_valid_frac_min):
            meas.append({"roi": name, "rect": [x, y, w, h], "skip": True, "reason": "low_valid_frac", "valid_frac": valid_frac})
            continue

        n_corners = shi_tomasi_count(pr, m, cfg)
        (dx, dy), resp = phase_corr_shift(pr, cr, cfg.use_hanning)

        corner_ratio = min(1.0, n_corners / max(1.0, float(cfg.st_min_corners)))
        weight = max(0.0, resp * corner_ratio * np.sqrt(valid_frac))

        meas.append({
            "roi": name, "rect": [x, y, w, h],
            "dx": float(dx), "dy": float(dy),
            "resp": float(resp), "corners": int(n_corners),
            "valid_frac": float(valid_frac), "weight": float(weight),
            "skip": False
        })

    usable = [m for m in meas if (not m["skip"]) and (m["resp"] >= cfg.roi_pc_resp_thresh)
              and (m["corners"] >= cfg.st_min_corners) and (m["weight"] > 0)]

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

    # weighted median
    def wmed(v: np.ndarray, w: np.ndarray) -> float:
        idx = np.argsort(v)
        v2, w2 = v[idx], w[idx]
        c = np.cumsum(w2)
        cut = 0.5 * float(c[-1])
        j = int(np.searchsorted(c, cut))
        return float(v2[min(j, len(v2) - 1)])

    dx0 = wmed(dxs, ws)
    dy0 = wmed(dys, ws)

    res = np.sqrt((dxs - dx0) ** 2 + (dys - dy0) ** 2)
    mad = float(np.median(np.abs(res - np.median(res))) + 1e-6)
    thr = float(cfg.mad_k) * (1.4826 * mad + 1e-6)

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

    dx = wmed(np.array([m["dx"] for m in inliers], np.float32),
              np.array([m["weight"] for m in inliers], np.float32))
    dy = wmed(np.array([m["dy"] for m in inliers], np.float32),
              np.array([m["weight"] for m in inliers], np.float32))
    dbg["dx_final"] = float(dx)
    dbg["dy_final"] = float(dy)
    dbg["resp_mean_inliers"] = float(np.mean([m["resp"] for m in inliers]))
    return (float(dx), float(dy)), dbg


def ecc_try(prev_f32: np.ndarray,
             curr_f32: np.ndarray,
             mask_u8: np.ndarray,
             init_T: np.ndarray,
             mode: EccMode,
             cfg: CamMotionConfig) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"mode": mode, "ok": False}

    if mode == "euclidean":
        motion = cv2.MOTION_EUCLIDEAN
    else:
        motion = cv2.MOTION_AFFINE  # Affine2

    warp = init_T.astype(np.float32).copy()

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(cfg.ecc_iterations),
                float(cfg.ecc_eps))
    try:
        cc, warp = cv2.findTransformECC(
            templateImage=curr_f32,
            inputImage=prev_f32,
            warpMatrix=warp,
            motionType=motion,
            criteria=criteria,
            inputMask=mask_u8
        )
        dbg["ecc_cc"] = float(cc)
        dbg["ok"] = True
        return warp, dbg
    except cv2.error as e:
        dbg["reason"] = "ecc_failed"
        dbg["cv2_error"] = str(e)[:200]
        return None, dbg


def max_corner_displacement(T_2x3: np.ndarray, H: int, W: int) -> float:
    pts = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = cv2.transform(pts, T_2x3.astype(np.float32))
    d = np.linalg.norm(pts2 - pts, axis=2).reshape(-1)
    return float(np.max(d)) if d.size else 0.0


def estimate_camera_motion(
    prev_feat: np.ndarray,
    curr_feat: np.ndarray,
    valid_mask: np.ndarray,
    prev_spec_mask: Optional[np.ndarray],
    curr_spec_mask: Optional[np.ndarray],
    cfg: CamMotionConfig,
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    Step2（统一技术路径）：
    - 输入建议 LoG（prev_feat/curr_feat），valid_mask 必须给（字幕 hard mask）
    - spec_mask 仅用于“权重抑制”（不再做任何反光/模糊替换链条）
    - Stage1/2：PhaseCorrelation（global + ROI consensus）
    - Stage3：ECC 回退：Euclidean(平移+旋转) -> Affine2
    """
    prev_u8 = ensure_gray_u8(prev_feat)
    curr_u8 = ensure_gray_u8(curr_feat)
    H, W = curr_u8.shape[:2]

    if valid_mask.shape != (H, W):
        raise ValueError("valid_mask shape mismatch")

    # hard invalid fill（字幕洞补齐）：复用 preprocessing 的公共逻辑
    prev_fill = apply_valid_mask_fill(prev_u8, valid_mask, sigma=3.0)
    curr_fill = apply_valid_mask_fill(curr_u8, valid_mask, sigma=3.0)

    # weight + err mask（核心：spec_mask 只走这里）
    w_f32, err_mask, wdbg = build_weight_and_err_mask(valid_mask, prev_spec_mask, curr_spec_mask, cfg)

    # phase-corr 输入：weighted zero-mean
    prev_use = weighted_zero_mean(prev_fill, w_f32)
    curr_use = weighted_zero_mean(curr_fill, w_f32)

    debug: Dict[str, Any] = {"method": "identity", "weighting": wdbg}

    # ---------- Stage 1: global phase correlation ----------
    (dx, dy), resp = phase_corr_shift(prev_use, curr_use, cfg.use_hanning)

    err0 = median_abs_error(curr_fill, prev_fill, err_mask)
    T_test = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    prev_warp_test = warp_u8(prev_fill, T_test)
    err1 = median_abs_error(curr_fill, prev_warp_test, err_mask)

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

    used_T: Optional[np.ndarray] = None
    global_ok = (
        resp >= cfg.global_pc_resp_thresh and
        abs(dx) <= cfg.max_abs_shift_px and abs(dy) <= cfg.max_abs_shift_px and
        (err1 / (err0 + 1e-6) <= cfg.global_err_ratio_thresh) and
        (err0 - err1 >= cfg.global_min_improve)
    )
    if global_ok:
        used_T = T_test
        debug["method"] = "global_phase_correlation"

    # ---------- Stage 2: ROI consensus ----------
    if used_T is None:
        shift2, roi_dbg = roi_consensus_shift(prev_use.astype(np.uint8), curr_use.astype(np.uint8), err_mask, cfg)
        debug["roi_consensus"] = roi_dbg
        if shift2 is not None:
            dx2, dy2 = shift2
            dx2 = float(np.clip(dx2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            dy2 = float(np.clip(dy2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            T2 = np.array([[1.0, 0.0, dx2], [0.0, 1.0, dy2]], dtype=np.float32)
            prev_warp2 = warp_u8(prev_fill, T2)
            err2 = median_abs_error(curr_fill, prev_warp2, err_mask)

            debug["roi_consensus_check"] = {
                "dx": float(dx2), "dy": float(dy2),
                "err_aligned": float(err2),
                "err_ratio": float(err2 / (err0 + 1e-6)),
                "improve": float(err0 - err2),
            }
            if (err2 / (err0 + 1e-6) <= cfg.global_err_ratio_thresh) and (err0 - err2 >= cfg.global_min_improve):
                used_T = T2
                debug["method"] = "roi_consensus_phase_correlation"

    # ---------- Stage 3: ECC fallback (EUCLIDEAN -> AFFINE) ----------
    if used_T is None and cfg.enable_ecc_fallback:
        # ECC 用 float32 [0,1]
        prev_f32 = (prev_fill.astype(np.float32) / 255.0)
        curr_f32 = (curr_fill.astype(np.float32) / 255.0)

        # init：用 global dx/dy 作为“还不错的初值”
        init = np.array([[1.0, 0.0, float(np.clip(dx, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))],
                         [0.0, 1.0, float(np.clip(dy, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))]], dtype=np.float32)

        ecc_dbg_all: List[Dict[str, Any]] = []
        for mode in cfg.ecc_order:
            warp, edbg = ecc_try(prev_f32, curr_f32, err_mask, init, mode, cfg)
            ecc_dbg_all.append(edbg)
            if warp is None:
                continue

            # clip translation
            warp = warp.astype(np.float32)
            warp[0, 2] = float(np.clip(warp[0, 2], -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            warp[1, 2] = float(np.clip(warp[1, 2], -cfg.max_abs_shift_px, cfg.max_abs_shift_px))

            prev_warp3 = warp_u8(prev_fill, warp)
            err3 = median_abs_error(curr_fill, prev_warp3, err_mask)

            edbg["err_aligned"] = float(err3)
            edbg["err_ratio"] = float(err3 / (err0 + 1e-6))
            edbg["improve"] = float(err0 - err3)

            if (err3 / (err0 + 1e-6) <= cfg.global_err_ratio_thresh) and (err0 - err3 >= cfg.global_min_improve):
                used_T = warp
                debug["method"] = f"ecc_{mode}"
                break

        debug["ecc"] = ecc_dbg_all

    # ---------- Final warp ----------
    if used_T is None:
        out = warp_src if warp_src is not None else prev_u8
        out = ensure_gray_u8(out)
        return CameraMotionResult(prev_aligned=out.copy(), T=None, camera_moving=False, debug=debug)

    src = warp_src if warp_src is not None else prev_u8
    src = ensure_gray_u8(src)
    prev_aligned = warp_u8(src, used_T)

    max_disp = max_corner_displacement(used_T, H, W)
    camera_moving = bool(max_disp >= float(cfg.moving_thresh_px))

    debug.update({
        "final_T": used_T.tolist(),
        "max_corner_displacement": float(max_disp),
        "camera_moving": bool(camera_moving),
    })
    return CameraMotionResult(prev_aligned=prev_aligned, T=used_T, camera_moving=camera_moving, debug=debug)


# convenience: run step2 from BGR directly
def estimate_camera_motion_from_bgr(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    feature: Literal["log", "intensity"] = "log",
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    pre_prev = preprocess_frame(prev_bgr, pre_cfg)
    pre_curr = preprocess_frame(curr_bgr, pre_cfg)

    prev_feat = pre_prev.log if feature == "log" else pre_prev.intensity
    curr_feat = pre_curr.log if feature == "log" else pre_curr.intensity

    src = warp_src if warp_src is not None else prev_feat

    # motion 用更稳的 valid：交集
    valid_use = cv2.bitwise_and(pre_prev.valid_mask, pre_curr.valid_mask)

    return estimate_camera_motion(
        prev_feat=prev_feat,
        curr_feat=curr_feat,
        valid_mask=valid_use,
        prev_spec_mask=pre_prev.spec_mask,
        curr_spec_mask=pre_curr.spec_mask,
        cfg=cam_cfg,
        warp_src=src,
    )
