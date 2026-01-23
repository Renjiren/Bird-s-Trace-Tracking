# camera_motion.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2


@dataclass
class CameraMotionResult:
    prev_aligned: np.ndarray                 # aligned prev frame to current
    T: Optional[np.ndarray]                  # 3x3 homography or 2x3 affine
    model: str                               # "homography" or "affine" or "none"
    camera_moving: bool
    debug: Dict[str, Any]


def _ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    """Ensure grayscale uint8 image."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _apply_mask(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    If mask provided, it should be uint8 with:
    - 255 for valid pixels to use
    - 0 for masked-out (e.g., subtitles region)
    """
    if mask is None:
        return img
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=mask)


def estimate_and_compensate_camera_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    model: str = "homography",   # "homography" or "affine"
    # ORB params
    nfeatures: int = 1000,
    fast_threshold: int = 20,
    # matching/filtering
    top_match_percent: float = 0.25,  # keep best X% matches (0~1)
    min_matches: int = 30,
    # RANSAC params
    ransac_reproj_threshold: float = 3.0,
    ransac_confidence: float = 0.995,
    # moving decision
    moving_disp_median_thresh: float = 1.0,  # pixels
) -> CameraMotionResult:
    """
    Step 2:
    1) ORB detect+compute on prev/curr
    2) BFMatcher (Hamming) match
    3) RANSAC estimate transformation (homography or affine)
    4) Warp prev -> align to curr

    Inputs:
      prev_gray, curr_gray: grayscale uint8 images from Step1 (or will be converted)
      valid_mask: optional mask to ignore subtitles/overlays etc (255=valid, 0=ignore)

    Outputs:
      CameraMotionResult (prev_aligned, T, model, camera_moving, debug)
    """
    prev_gray = _ensure_gray_u8(prev_gray)
    curr_gray = _ensure_gray_u8(curr_gray)

    H, W = curr_gray.shape[:2]

    prev_use = _apply_mask(prev_gray, valid_mask)
    curr_use = _apply_mask(curr_gray, valid_mask)

    # 1) ORB features
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        fastThreshold=fast_threshold,
    )
    kp1, des1 = orb.detectAndCompute(prev_use, None)
    kp2, des2 = orb.detectAndCompute(curr_use, None)

    debug: Dict[str, Any] = {
        "kp_prev": 0 if kp1 is None else len(kp1),
        "kp_curr": 0 if kp2 is None else len(kp2),
        "matches_total": 0,
        "matches_kept": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "disp_median": None,
        "disp_mean": None,
    }

    # If not enough descriptors, return identity (no alignment)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return CameraMotionResult(
            prev_aligned=prev_gray.copy(),
            T=None,
            model="none",
            camera_moving=False,
            debug={**debug, "reason": "not_enough_keypoints_or_descriptors"},
        )

    # 2) Match descriptors (tutorial-style BFMatcher + Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    debug["matches_total"] = len(matches)

    if len(matches) < min_matches:
        return CameraMotionResult(
            prev_aligned=prev_gray.copy(),
            T=None,
            model="none",
            camera_moving=False,
            debug={**debug, "reason": "not_enough_matches"},
        )

    # Keep best X% matches to reduce outliers (common practical trick)
    keep_n = max(min_matches, int(len(matches) * float(top_match_percent)))
    matches_kept = matches[:keep_n]
    debug["matches_kept"] = len(matches_kept)

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches_kept]).reshape(-1, 1, 2)  # prev
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches_kept]).reshape(-1, 1, 2)  # curr

    # 3) RANSAC estimate transformation
    T = None
    inlier_mask = None
    used_model = model

    if model == "homography":
        # Need >= 4
        if len(pts1) >= 4:
            T, inlier_mask = cv2.findHomography(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
                confidence=ransac_confidence,
            )
        if T is None or inlier_mask is None:
            # fallback to affine
            used_model = "affine"

    if used_model == "affine":
        # estimateAffinePartial2D is robust and often enough for moderate camera motion
        # (translation + rotation + scale)
        # Needs >= 3
        if len(pts1) >= 3:
            T_aff, inlier_mask = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
                confidence=ransac_confidence,
                refineIters=10,
            )
            T = T_aff  # 2x3
        if T is None or inlier_mask is None:
            return CameraMotionResult(
                prev_aligned=prev_gray.copy(),
                T=None,
                model="none",
                camera_moving=False,
                debug={**debug, "reason": "ransac_failed"},
            )

    inlier_mask = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    debug["inliers"] = inliers
    debug["inlier_ratio"] = float(inliers) / float(len(inlier_mask) + 1e-9)

    # 3.1) Compute camera motion magnitude from inlier displacements (pixels)
    pts1_in = pts1[inlier_mask][:, 0, :]  # (N,2)
    pts2_in = pts2[inlier_mask][:, 0, :]
    if len(pts1_in) >= 5:
        disp = np.linalg.norm(pts2_in - pts1_in, axis=1)
        debug["disp_median"] = float(np.median(disp))
        debug["disp_mean"] = float(np.mean(disp))
        camera_moving = (debug["disp_median"] >= float(moving_disp_median_thresh))
    else:
        camera_moving = False

    # 4) Warp prev frame to current frame coordinates
    if used_model == "homography":
        # 3x3
        prev_aligned = cv2.warpPerspective(
            prev_gray, T, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        # affine 2x3
        prev_aligned = cv2.warpAffine(
            prev_gray, T, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return CameraMotionResult(
        prev_aligned=prev_aligned,
        T=T,
        model=used_model,
        camera_moving=camera_moving,
        debug=debug,
    )
