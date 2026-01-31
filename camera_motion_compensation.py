# camera_motion_compensation.py
# New Step2: PC + ECC

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

import numpy as np
import cv2

from preprocessing import (
    preprocess_frame,
    PreprocessConfig,
    ensure_gray_u8,
)


# Config
@dataclass(frozen=True)
class CamMotionConfig:
    # -------- Phase Correlation --------
    use_hanning: bool = True
    pc_resp_thresh: float = 0.15
    max_abs_shift_px: float = 50.0

    # -------- ECC fallback --------
    enable_ecc: bool = True
    ecc_iterations: int = 50
    ecc_eps: float = 1e-4

    # -------- decision --------
    moving_thresh_px: float = 1.0

    # -------- hard invalid fill (subtitle) --------
    fill_sigma: float = 3.0


@dataclass
class CameraMotionResult:
    prev_aligned: np.ndarray
    T: Optional[np.ndarray]          # 2x3 translation matrix
    camera_moving: bool
    debug: Dict[str, Any]

# Utils
def _hard_fill_invalid(
    img_u8: np.ndarray,
    valid_mask: Optional[np.ndarray],
    sigma: float,
) -> np.ndarray:
    """
    用于 Step2（仅用于对齐）：
    防止 Phase Correlation / ECC 看到“黑洞字幕”。
    """
    if valid_mask is None:
        return img_u8
    if not np.any(valid_mask == 0):
        return img_u8

    blur = cv2.GaussianBlur(
        img_u8,
        (0, 0),
        sigmaX=float(max(0.0, sigma)),
        sigmaY=float(max(0.0, sigma)),
    )
    out = img_u8.copy()
    out[valid_mask == 0] = blur[valid_mask == 0]
    return out


def _phase_correlation(
    a_u8: np.ndarray,
    b_u8: np.ndarray,
    use_hanning: bool,
) -> tuple[float, float, float]:
    """
    Returns dx, dy, resp
    """
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)

    if use_hanning:
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    else:
        (dx, dy), resp = cv2.phaseCorrelate(a, b)

    return float(dx), float(dy), float(resp)


def _ecc_translation(
    prev_u8: np.ndarray,
    curr_u8: np.ndarray,
    valid_mask: Optional[np.ndarray],
    cfg: CamMotionConfig,
) -> Optional[tuple[float, float]]:
    """
    ECC translation-only fallback.
    """
    try:
        template = curr_u8.astype(np.float32) / 255.0
        inp = prev_u8.astype(np.float32) / 255.0

        warp = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(cfg.ecc_iterations),
            float(cfg.ecc_eps),
        )

        cc, warp = cv2.findTransformECC(
            templateImage=template,
            inputImage=inp,
            warpMatrix=warp,
            motionType=cv2.MOTION_TRANSLATION,
            criteria=criteria,
            inputMask=valid_mask,
        )

        dx = float(warp[0, 2])
        dy = float(warp[1, 2])
        return dx, dy
    except cv2.error:
        return None


# Main fonction
def estimate_camera_translation(
    prev_feat: np.ndarray,
    curr_feat: np.ndarray,
    valid_mask: Optional[np.ndarray],
    cfg: CamMotionConfig,
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    Estimate global translation prev -> curr.
    Recommended feature: LoG(I_norm)
    """

    prev_feat = ensure_gray_u8(prev_feat)
    curr_feat = ensure_gray_u8(curr_feat)
    H, W = curr_feat.shape

    debug: Dict[str, Any] = {}

    # -------- hard fill invalid (subtitle) --------
    prev_use = _hard_fill_invalid(prev_feat, valid_mask, cfg.fill_sigma)
    curr_use = _hard_fill_invalid(curr_feat, valid_mask, cfg.fill_sigma)

    # -------- Phase Correlation --------
    dx, dy, resp = _phase_correlation(prev_use, curr_use, cfg.use_hanning)

    debug["pc"] = {
        "dx": dx,
        "dy": dy,
        "resp": resp,
        "resp_thresh": cfg.pc_resp_thresh,
    }

    use_dxdy: Optional[tuple[float, float]] = None

    if (
        resp >= cfg.pc_resp_thresh
        and abs(dx) <= cfg.max_abs_shift_px
        and abs(dy) <= cfg.max_abs_shift_px
    ):
        use_dxdy = (dx, dy)
        debug["method"] = "phase_correlation"

    # -------- ECC fallback --------
    if use_dxdy is None and cfg.enable_ecc:
        ecc_shift = _ecc_translation(prev_use, curr_use, valid_mask, cfg)
        if ecc_shift is not None:
            dx2, dy2 = ecc_shift
            if abs(dx2) <= cfg.max_abs_shift_px and abs(dy2) <= cfg.max_abs_shift_px:
                use_dxdy = (dx2, dy2)
                debug["method"] = "ecc_translation"
                debug["ecc"] = {"dx": dx2, "dy": dy2}
            else:
                debug["ecc"] = "shift_too_large"
        else:
            debug["ecc"] = "failed"

    # -------- no reliable alignment --------
    if use_dxdy is None:
        src = warp_src if warp_src is not None else prev_feat
        return CameraMotionResult(
            prev_aligned=ensure_gray_u8(src).copy(),
            T=None,
            camera_moving=False,
            debug=debug,
        )

    # -------- warp --------
    dx_f, dy_f = use_dxdy
    T = np.array(
        [[1.0, 0.0, dx_f],
         [0.0, 1.0, dy_f]],
        dtype=np.float32,
    )

    src = warp_src if warp_src is not None else prev_feat
    src = ensure_gray_u8(src)

    prev_aligned = cv2.warpAffine(
        src,
        T,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    shift_norm = float(np.hypot(dx_f, dy_f))
    camera_moving = bool(shift_norm >= cfg.moving_thresh_px)

    debug["final"] = {
        "dx": dx_f,
        "dy": dy_f,
        "shift_norm": shift_norm,
        "camera_moving": camera_moving,
    }

    return CameraMotionResult(
        prev_aligned=prev_aligned,
        T=T,
        camera_moving=camera_moving,
        debug=debug,
    )


# Convenience wrapper
def estimate_camera_translation_from_bgr(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    feature: Literal["log"] = "log",
    warp_src: Optional[np.ndarray] = None,
) -> CameraMotionResult:
    """
    One-call Step2 entry.
    """
    pre_prev = preprocess_frame(prev_bgr, pre_cfg)
    pre_curr = preprocess_frame(curr_bgr, pre_cfg)

    if feature == "log":
        prev_feat = pre_prev.log
        curr_feat = pre_curr.log
    else:
        raise ValueError("Only 'log' feature is supported in new Step2.")

    src = warp_src if warp_src is not None else prev_feat

    return estimate_camera_translation(
        prev_feat=prev_feat,
        curr_feat=curr_feat,
        valid_mask=pre_prev.valid_mask,
        cfg=cam_cfg,
        warp_src=src,
    )
