# Camera motion compensation 
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Literal
import numpy as np
import cv2

from preprocessing import ensure_gray_u8, preprocess_frame, PreprocessConfig, apply_valid_mask_fill

RoiMode = Literal["corners", "strips", "corners+strips"]


@dataclass(frozen=True)
class CamMotionConfig:
    # ---------- ROI selection ----------
    roi_mode: RoiMode = "strips"
    roi_frac: float = 0.35
    strip_frac: float = 0.18
    margin_frac: float = 0.02
    use_hanning: bool = True

    # ROI valid pixel ratio minimum
    roi_valid_frac_min: float = 0.45

    # ---------- Acceptance thresholds (shared by PC/ROI/ECC) ----------
    max_abs_shift_px: float = 50.0
    moving_thresh_px: float = 1.0

    # Phase correlation response thresholds
    global_pc_resp_thresh: float = 0.30
    roi_pc_resp_thresh: float = 0.22

    # Error-based acceptance (scheme A: mean abs error + robust handling)
    err_ratio_thresh: float = 0.92
    min_improve: float = 0.5

    # When err0 is too small, "improve" and "ratio" become meaningless;
    # in that case we relax acceptance to rely more on correlation confidence.
    err0_small: float = 1.0

    # ---------- ROI consensus details ----------
    st_max_corners: int = 300
    st_quality: float = 0.01
    st_min_distance: int = 7
    st_block_size: int = 7
    st_min_corners: int = 40

    mad_k: float = 3.0
    min_inlier_rois: int = 3

    # ---------- ECC fallback ----------
    enable_ecc_fallback: bool = True
    ecc_iterations: int = 60
    ecc_eps: float = 1e-4
    ecc_cc_thresh: float = 0.40   # optional ECC correlation threshold

    # ---------- Soft spec weighting ----------
    use_soft_spec_weight: bool = True
    specular_weight: float = 0.35  # weight applied to error inside spec region (0..1)

    save_roi_infos: bool = False


@dataclass
class CameraMotionResult:
    prev_aligned: np.ndarray
    T: Optional[np.ndarray]          # 2x3 warp matrix
    camera_moving: bool
    debug: Dict[str, Any]


#--------------- utils ---------------
def make_rois(H: int, W: int, cfg: CamMotionConfig) -> List[Tuple[int, int, int, int, str]]:
    """Generate ROIs for robust consensus estimation."""
    mx = int(W * cfg.margin_frac)
    my = int(H * cfg.margin_frac)

    def clamp_rect(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
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


def phase_corr_shift(a_u8: np.ndarray, b_u8: np.ndarray, use_hanning: bool) -> Tuple[Tuple[float, float], float]:
    """Phase correlation shift between two uint8 gray images."""
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)

    if use_hanning:
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        shift, resp = cv2.phaseCorrelate(a, b, win)
    else:
        shift, resp = cv2.phaseCorrelate(a, b)

    return (float(shift[0]), float(shift[1])), float(resp)


def warp_u8(img_u8: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Warp with translation (dx, dy)."""
    H, W = img_u8.shape[:2]
    T = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img_u8, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build_spec_union_bad(prev_spec_mask: Optional[np.ndarray],
                          curr_spec_mask: Optional[np.ndarray],
                          H: int,
                          W: int) -> Optional[np.ndarray]:
    """
    Build a boolean 'spec_bad' map:
      True  -> specular-like (spec_mask==0)
      False -> normal
    We use union across prev/curr if both exist (more robust).
    """
    if prev_spec_mask is None and curr_spec_mask is None:
        return None

    if prev_spec_mask is None:
        bad = (curr_spec_mask == 0)
    elif curr_spec_mask is None:
        bad = (prev_spec_mask == 0)
    else:
        bad = (prev_spec_mask == 0) | (curr_spec_mask == 0)

    if bad.shape != (H, W):
        # best-effort shape fix (avoid hard crash)
        bad = cv2.resize(bad.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    return bad.astype(bool)


def weighted_mean_abs_error(curr_u8: np.ndarray,
                             prev_u8: np.ndarray,
                             valid_mask: Optional[np.ndarray],
                             spec_bad: Optional[np.ndarray],
                             cfg: CamMotionConfig) -> float:
    """
    Scheme-A error metric:
      mean(|curr-prev|) computed only on valid pixels,
      and down-weighted inside specular-like pixels.
    This matches Step3's idea: "soft spec weighting" instead of hard ignore.
    """
    diff = cv2.absdiff(curr_u8, prev_u8).astype(np.float32)

    if valid_mask is not None:
        v = (valid_mask > 0)
    else:
        v = np.ones(diff.shape, dtype=bool)

    if not np.any(v):
        return float(np.mean(diff))

    if cfg.use_soft_spec_weight and spec_bad is not None:
        w = float(np.clip(cfg.specular_weight, 0.0, 1.0))
        if w < 0.999:
            s = spec_bad & v
            diff[s] *= w

    vals = diff[v]
    if vals.size == 0:
        return float(np.mean(diff))
    return float(np.mean(vals))


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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    c = np.cumsum(w)
    cutoff = 0.5 * float(c[-1])
    j = int(np.searchsorted(c, cutoff))
    return float(v[min(j, len(v) - 1)])


def roi_consensus_shift(prev_u8: np.ndarray,
                         curr_u8: np.ndarray,
                         valid_mask: Optional[np.ndarray],
                         cfg: CamMotionConfig) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    """
    Robust shift estimation:
    - phase correlation in multiple ROIs
    - weight by (resp * corner_ratio * sqrt(valid_frac))
    - MAD-based outlier rejection
    """
    H, W = curr_u8.shape[:2]
    rois = make_rois(H, W, cfg)

    meas: List[Dict[str, Any]] = []
    for (x, y, w, h, name) in rois:
        pr = prev_u8[y:y+h, x:x+w]
        cr = curr_u8[y:y+h, x:x+w]

        if valid_mask is not None and valid_mask.shape == (H, W):
            m = valid_mask[y:y+h, x:x+w]
            valid_frac = float(np.mean(m > 0))
            if valid_frac < float(cfg.roi_valid_frac_min):
                meas.append({"roi": name, "rect": [x, y, w, h], "skip": True, "reason": "low_valid_frac", "valid_frac": valid_frac})
                continue
            m_use = m
        else:
            valid_frac = 1.0
            m_use = None

        n_corners = shi_tomasi_count(pr, m_use, cfg)
        (dx, dy), resp = phase_corr_shift(pr, cr, cfg.use_hanning)

        corner_ratio = min(1.0, n_corners / max(1.0, float(cfg.st_min_corners)))
        weight = max(0.0, resp * corner_ratio * np.sqrt(valid_frac))

        meas.append({
            "roi": name,
            "rect": [int(x), int(y), int(w), int(h)],
            "dx": float(dx), "dy": float(dy),
            "resp": float(resp),
            "corners": int(n_corners),
            "valid_frac": float(valid_frac),
            "weight": float(weight),
            "skip": False,
        })

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

    dx0 = weighted_median(dxs, ws)
    dy0 = weighted_median(dys, ws)

    res = np.sqrt((dxs - dx0) ** 2 + (dys - dy0) ** 2)
    mad = float(np.median(np.abs(res - np.median(res))) + 1e-6)
    thr = cfg.mad_k * (1.4826 * mad + 1e-6)

    inlier = res <= max(thr, 1.0)
    inliers = [u for u, keep in zip(usable, inlier.tolist()) if keep]

    dbg.update({
        "dx0": float(dx0),
        "dy0": float(dy0),
        "mad": float(mad),
        "inliers": int(len(inliers)),
        "inlier_thr": float(max(thr, 1.0)),
    })

    if len(inliers) < cfg.min_inlier_rois:
        dbg["reason"] = "not_enough_inliers_after_mad"
        return None, dbg

    dxs2 = np.array([m["dx"] for m in inliers], dtype=np.float32)
    dys2 = np.array([m["dy"] for m in inliers], dtype=np.float32)
    ws2 = np.array([m["weight"] for m in inliers], dtype=np.float32)

    dx = weighted_median(dxs2, ws2)
    dy = weighted_median(dys2, ws2)

    dbg.update({
        "dx_final": float(dx),
        "dy_final": float(dy),
        "resp_mean_inliers": float(np.mean([m["resp"] for m in inliers])),
    })

    return (float(dx), float(dy)), dbg


def ecc_translation(prev_u8: np.ndarray,
                     curr_u8: np.ndarray,
                     valid_mask: Optional[np.ndarray],
                     cfg: CamMotionConfig,
                     init_shift: Optional[Tuple[float, float]] = None) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    """
    ECC translation (masked). IMPORTANT:
    - We also return debug and let the caller apply the same acceptance logic.
    - Use init_shift from PC/ROI as initialization for better convergence.
    """
    dbg: Dict[str, Any] = {"ecc_enabled": True}
    try:
        template = curr_u8.astype(np.float32) / 255.0
        inp = prev_u8.astype(np.float32) / 255.0

        warp = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)
        if init_shift is not None:
            warp[0, 2] = float(init_shift[0])
            warp[1, 2] = float(init_shift[1])
            dbg["init_dx"] = float(init_shift[0])
            dbg["init_dy"] = float(init_shift[1])

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(cfg.ecc_iterations),
                    float(cfg.ecc_eps))

        cc, warp = cv2.findTransformECC(
            templateImage=template,
            inputImage=inp,
            warpMatrix=warp,
            motionType=cv2.MOTION_TRANSLATION,
            criteria=criteria,
            inputMask=valid_mask
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


def accept_candidate(resp_like: float,
                      resp_thresh: float,
                      dx: float, dy: float,
                      err0: float, err1: float,
                      cfg: CamMotionConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    Shared acceptance logic for PC/ROI/ECC.
    When err0 is tiny, error improvement is unreliable, so we relax to resp-only gate.
    """
    info: Dict[str, Any] = {}

    if abs(dx) > cfg.max_abs_shift_px or abs(dy) > cfg.max_abs_shift_px:
        info["reject_reason"] = "shift_too_large"
        return False, info

    ratio = float(err1 / (err0 + 1e-6))
    improve = float(err0 - err1)

    info.update({"err0": float(err0), "err1": float(err1), "ratio": ratio, "improve": improve})

    # If baseline error is too small, rely mainly on correlation confidence
    if err0 <= float(cfg.err0_small):
        ok = (resp_like >= resp_thresh)
        info["mode"] = "err0_small_relaxed"
        info["resp_like"] = float(resp_like)
        info["resp_thresh"] = float(resp_thresh)
        if not ok:
            info["reject_reason"] = "low_resp_like"
        return ok, info

    # Normal gate: resp + error improvements
    ok = (
        (resp_like >= resp_thresh) and
        (ratio <= float(cfg.err_ratio_thresh)) and
        (improve >= float(cfg.min_improve))
    )
    info["mode"] = "normal"
    info["resp_like"] = float(resp_like)
    info["resp_thresh"] = float(resp_thresh)
    info["err_ratio_thresh"] = float(cfg.err_ratio_thresh)
    info["min_improve"] = float(cfg.min_improve)

    if not ok:
        info["reject_reason"] = "resp_or_error_gate_failed"
    return ok, info


# ------------- main function ---------------
def estimate_camera_translation(
    prev_feat: np.ndarray,
    curr_feat: np.ndarray,
    valid_mask: Optional[np.ndarray],            # current-frame hard mask (255 valid / 0 invalid)
    prev_spec_mask: Optional[np.ndarray],        # 255 normal / 0 spec-like
    curr_spec_mask: Optional[np.ndarray],        # 255 normal / 0 spec-like
    cfg: CamMotionConfig,
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    Estimate translation that warps prev -> curr.

    Key design (aligned with your Step3):
    - Hard subtitle mask is handled by apply_valid_mask_fill() (no hard 0 holes).
    - Spec is NOT hard-ignored; it is down-weighted only in the error metric (acceptance).
    - Error metric uses mean absolute error (scheme A), which is much more stable than median on LoG-u8.
    - ECC is accepted/rejected using the same gates as PC/ROI, and initialized from the best PC/ROI guess.
    """
    prev_feat = ensure_gray_u8(prev_feat)
    curr_feat = ensure_gray_u8(curr_feat)
    H, W = curr_feat.shape[:2]

    # ----- build hard mask -----
    hard_mask: Optional[np.ndarray]
    if valid_mask is None:
        hard_mask = None
    else:
        hard_mask = valid_mask.astype(np.uint8)
        # normalize to 0/255 in case it is 0/1
        if hard_mask.max() <= 1:
            hard_mask = (hard_mask * 255).astype(np.uint8)

    # ----- build spec union (soft) -----
    spec_bad = build_spec_union_bad(prev_spec_mask, curr_spec_mask, H, W)

    # ----- apply Step3-style hard mask fill (avoid edges/holes) -----
    prev_use = apply_valid_mask_fill(prev_feat, hard_mask, sigma=3.0)
    curr_use = apply_valid_mask_fill(curr_feat, hard_mask, sigma=3.0)

    debug: Dict[str, Any] = {
        "hard_mask": None if hard_mask is None else {
            "valid_ratio": float(np.mean(hard_mask > 0)),
            "invalid_ratio": float(np.mean(hard_mask == 0)),
        },
        "spec_union": None if spec_bad is None else {
            "spec_ratio_valid": float(np.mean(spec_bad[hard_mask > 0])) if hard_mask is not None and np.any(hard_mask > 0) else float(np.mean(spec_bad))
        },
        "method": "identity",
    }

    # Precompute baseline error (unaligned)
    err0 = weighted_mean_abs_error(curr_use, prev_use, hard_mask, spec_bad, cfg)


    # --------------- Stage 1: global PC ---------------
    (dx1, dy1), resp1 = phase_corr_shift(prev_use, curr_use, cfg.use_hanning)
    prev_w1 = warp_u8(prev_use, dx1, dy1)
    err1 = weighted_mean_abs_error(curr_use, prev_w1, hard_mask, spec_bad, cfg)

    ok1, ok1_info = accept_candidate(resp1, cfg.global_pc_resp_thresh, dx1, dy1, err0, err1, cfg)
    debug["global_pc"] = {
        "dx": float(dx1), "dy": float(dy1), "resp": float(resp1),
        **ok1_info
    }

    best_shift: Optional[Tuple[float, float]] = None
    best_err: float = float("inf")
    best_tag: str = "identity"

    if ok1 and err1 < best_err:
        best_shift = (float(dx1), float(dy1))
        best_err = float(err1)
        best_tag = "global_phase_correlation"

    # ------------- Stage 2: ROI consensus ---------------
    shift2, roi_dbg = roi_consensus_shift(prev_use, curr_use, hard_mask, cfg)
    debug["roi_consensus"] = roi_dbg

    if shift2 is not None:
        dx2, dy2 = shift2
        dx2 = float(np.clip(dx2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
        dy2 = float(np.clip(dy2, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))

        prev_w2 = warp_u8(prev_use, dx2, dy2)
        err2 = weighted_mean_abs_error(curr_use, prev_w2, hard_mask, spec_bad, cfg)

        # ROI uses resp threshold from cfg.roi_pc_resp_thresh, but ROI itself has multiple resps;
        # we use the mean-inlier response as "resp_like" when available.
        resp2_like = float(roi_dbg.get("resp_mean_inliers", 0.0))
        ok2, ok2_info = accept_candidate(resp2_like, cfg.roi_pc_resp_thresh, dx2, dy2, err0, err2, cfg)

        debug["roi_check"] = {
            "dx": float(dx2), "dy": float(dy2),
            "resp_like": float(resp2_like),
            **ok2_info
        }

        if ok2 and err2 < best_err:
            best_shift = (dx2, dy2)
            best_err = float(err2)
            best_tag = "roi_consensus_phase_correlation"


    # ----------------- Stage 3: ECC fallback -----------------
    if cfg.enable_ecc_fallback:
        # Initialization: use the best PC/ROI guess (even if not accepted),
        # because it often helps ECC converge.
        init = None
        if shift2 is not None:
            init = (float(np.clip(shift2[0], -cfg.max_abs_shift_px, cfg.max_abs_shift_px)),
                    float(np.clip(shift2[1], -cfg.max_abs_shift_px, cfg.max_abs_shift_px)))
        else:
            init = (float(np.clip(dx1, -cfg.max_abs_shift_px, cfg.max_abs_shift_px)),
                    float(np.clip(dy1, -cfg.max_abs_shift_px, cfg.max_abs_shift_px)))

        ecc_shift, ecc_dbg = ecc_translation(prev_use, curr_use, hard_mask, cfg, init_shift=init)
        debug["ecc"] = ecc_dbg

        if ecc_shift is not None:
            dx3, dy3 = ecc_shift
            dx3 = float(np.clip(dx3, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))
            dy3 = float(np.clip(dy3, -cfg.max_abs_shift_px, cfg.max_abs_shift_px))

            prev_w3 = warp_u8(prev_use, dx3, dy3)
            err3 = weighted_mean_abs_error(curr_use, prev_w3, hard_mask, spec_bad, cfg)

            # ECC has "ecc_cc" as resp-like score
            cc = float(ecc_dbg.get("ecc_cc", 0.0))
            # Optionally require ecc_cc above a threshold (and also apply the same error gates)
            ok3, ok3_info = accept_candidate(cc, cfg.ecc_cc_thresh, dx3, dy3, err0, err3, cfg)

            debug["ecc_check"] = {
                "dx": float(dx3), "dy": float(dy3),
                **ok3_info
            }

            if ok3 and err3 < best_err:
                best_shift = (dx3, dy3)
                best_err = float(err3)
                best_tag = "ecc_translation"


    # ------------ Finalize ------------
    if best_shift is None:
        # no valid candidate accepted -> identity
        out = warp_src if warp_src is not None else prev_feat
        out = ensure_gray_u8(out)
        debug["method"] = "identity"
        return CameraMotionResult(prev_aligned=out.copy(), T=None, camera_moving=False, debug=debug)

    dx_f, dy_f = best_shift
    T = np.array([[1.0, 0.0, dx_f],
                  [0.0, 1.0, dy_f]], dtype=np.float32)

    src = warp_src if warp_src is not None else prev_feat
    src = ensure_gray_u8(src)
    prev_aligned = cv2.warpAffine(src, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    shift_norm = float(np.hypot(dx_f, dy_f))
    camera_moving = bool(shift_norm >= float(cfg.moving_thresh_px))

    debug.update({
        "method": best_tag,
        "final_dx": float(dx_f),
        "final_dy": float(dy_f),
        "shift_norm": float(shift_norm),
        "camera_moving": camera_moving,
        "err0_unaligned": float(err0),
        "best_err_aligned": float(best_err),
    })

    return CameraMotionResult(prev_aligned=prev_aligned, T=T, camera_moving=camera_moving, debug=debug)


def estimate_camera_translation_from_bgr(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    feature: Literal["log", "intensity"] = "log",
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    Recommended entry point:
    - feature="log" for estimating motion (robust), then warp intensity as warp_src if needed.
    - IMPORTANT: we use CURRENT-FRAME valid_mask (subtitle mask) to match Step3 behavior.
    """
    pre_prev = preprocess_frame(prev_bgr, pre_cfg)
    pre_curr = preprocess_frame(curr_bgr, pre_cfg)

    if feature == "log":
        prev_feat = pre_prev.log
        curr_feat = pre_curr.log
    else:
        prev_feat = pre_prev.intensity
        curr_feat = pre_curr.intensity

    # Warp source defaults to the same feature if not provided
    src = warp_src if warp_src is not None else prev_feat

    return estimate_camera_translation(
        prev_feat=prev_feat,
        curr_feat=curr_feat,
        valid_mask=pre_curr.valid_mask,        # <-- current frame hard subtitle mask
        prev_spec_mask=pre_prev.spec_mask,     # for spec union (soft)
        curr_spec_mask=pre_curr.spec_mask,
        cfg=cam_cfg,
        warp_src=src,
    )