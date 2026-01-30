# birds_pipeline.py
from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from preprocessing import (
    PreprocessConfig,
    preprocess_frame,
    ensure_bgr_u8,
    glare_stats_one_frame,
    adapt_glare_params_from_stats,
)
from camera_motion_compensation import CamMotionConfig, estimate_camera_translation
from candidate_generation import (
    CandidateGenConfig,
    MotionCandidateGenerator,
    generate_motion_candidates,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")

EG_VIDEOS = [
    "Ac4002", "Ac4003", "An3004", "An3013",
    "An6011", "Ci2001", "Ci3001", "Pa1003",
    "Gr5009", "Su2001", "Su2002", "Su2005"
]


# -------------------------
# IO + small helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)


def infer_frame_id(filename: str) -> int:
    stem, _ = os.path.splitext(os.path.basename(filename))
    try:
        return int(stem)
    except Exception:
        return -1


def list_videos(data_root: str) -> List[str]:
    return [v for v in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, v))]


def list_frames(video_dir: str) -> List[str]:
    files = [fn for fn in os.listdir(video_dir) if is_image(fn)]
    files.sort(key=lambda x: (infer_frame_id(x) < 0, infer_frame_id(x), x))
    return [os.path.join(video_dir, fn) for fn in files]


def select_videos(data_root: str, video_set: str, only_videos: Optional[List[str]]) -> List[str]:
    videos = list_videos(data_root)
    if only_videos:
        allow = set(only_videos)
        return [v for v in videos if v in allow]

    if video_set == "eg":
        allow = set(EG_VIDEOS)
        return [v for v in videos if v in allow]
    if video_set == "all":
        return videos
    raise ValueError(f"Unknown video_set: {video_set}")


def imread_bgr(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def imwrite(path: str, img: np.ndarray, overwrite: bool) -> None:
    if (not overwrite) and os.path.exists(path):
        return
    cv2.imwrite(path, img)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# visualization helpers
# -------------------------
def overlay_mask(bgr: np.ndarray, mask: np.ndarray, bgr_color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    """
    mask: uint8, 255=keep, 0=paint
    """
    base = bgr.copy()
    paint = base.copy()
    paint[mask == 0] = bgr_color
    return cv2.addWeighted(paint, float(alpha), base, 1.0 - float(alpha), 0.0)


def draw_dxdy_tag(
    bgr: np.ndarray,
    dx: float,
    dy: float,
    method: str,
    title: str,
    extra: Optional[str] = None,
) -> np.ndarray:
    img = bgr.copy()
    text1 = f"{title}  dx={dx:+.2f}  dy={dy:+.2f}"
    text2 = method if not extra else f"{method}  |  {extra}"

    x0, y0 = 10, 10
    w, h = 720, 70
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), thickness=-1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0.0)

    cv2.putText(img, text1, (x0 + 10, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text2, (x0 + 10, y0 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 210, 210), 2, cv2.LINE_AA)
    return img


def draw_boxes_on_bgr(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], max_draw: int = 80) -> np.ndarray:
    img = bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes[:max_draw]):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(i), (x, max(0, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def make_diff_vis(curr_u8: np.ndarray, prev_aligned_u8: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(curr_u8, prev_aligned_u8)
    p1 = float(np.percentile(diff, 1.0))
    p99 = float(np.percentile(diff, 99.0))
    denom = max(1e-6, p99 - p1)
    diff_n = np.clip((diff.astype(np.float32) - p1) / denom, 0.0, 1.0)
    diff_u8 = (diff_n * 255.0).astype(np.uint8)
    return cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)


def warp_u8(img_u8: np.ndarray, T_2x3: Optional[np.ndarray]) -> np.ndarray:
    if T_2x3 is None:
        return img_u8
    H, W = img_u8.shape[:2]
    return cv2.warpAffine(img_u8, T_2x3.astype(np.float32), (W, H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# -------------------------
# mask merge helpers
# -------------------------
def merge_valid_intersection(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """valid_mask: 255 valid / 0 invalid; intersection is AND."""
    if a is None:
        return b
    if b is None:
        return a
    if a.shape != b.shape:
        return b
    return cv2.bitwise_and(a, b)


def merge_spec_union_bad(prev_spec: Optional[np.ndarray], curr_spec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    spec_mask: 255 normal / 0 spec-like
    Want union of BAD (0): any frame bad -> bad
    For 0/255 masks: union_bad == bitwise_and.
    """
    if prev_spec is None:
        return curr_spec
    if curr_spec is None:
        return prev_spec
    if prev_spec.shape != curr_spec.shape:
        return curr_spec
    return cv2.bitwise_and(prev_spec, curr_spec)


# -------------------------
# stats helper (step1 tuning)
# -------------------------
def compute_pre_stats(pre) -> Dict[str, Any]:
    valid = (pre.valid_mask > 0)
    H, W = pre.intensity.shape[:2]
    if not np.any(valid):
        return {"H": int(H), "W": int(W), "valid_ratio": 0.0, "invalid_ratio": 0.0,
                "spec_ratio_valid": 0.0, "edge_density": 0.0,
                "smooth_gray_mean": 0.0, "smooth_gray_std": 0.0}

    return {
        "H": int(H),
        "W": int(W),
        "valid_ratio": float(np.mean(valid)),
        "invalid_ratio": float(np.mean(pre.valid_mask == 0)),
        "spec_ratio_valid": float(np.mean((pre.spec_mask == 0)[valid])),
        "edge_density": float(np.mean((pre.log >= 35)[valid])),
        "smooth_gray_mean": float(np.mean(pre.intensity[valid])),
        "smooth_gray_std": float(np.std(pre.intensity[valid])),
    }


def mean_key(frames_summary: List[Dict[str, Any]], key: str) -> float:
    xs: List[float] = []
    for fr in frames_summary:
        st = fr.get("stats", {})
        if key in st:
            xs.append(float(st[key]))
    return float(np.mean(xs)) if xs else 0.0


# -------------------------
# glare adapt helper (shared)
# -------------------------
def adapt_glare_for_video(frames: List[str], pre_cfg: PreprocessConfig, first_n: int) -> Tuple[PreprocessConfig, Dict[str, Any]]:
    """
    Per-video adapt glare params (do NOT change subtitle params).
    Returns (cfg_video, adapt_debug).
    """
    take_n = min(int(first_n), len(frames))
    if take_n <= 0:
        return pre_cfg, {"used": False, "reason": "first_n<=0"}

    glare_stats_list: List[Dict[str, float]] = []
    edge_list: List[float] = []

    for i in range(take_n):
        bgr = imread_bgr(frames[i])
        if bgr is None:
            continue
        pre_tmp = preprocess_frame(bgr, pre_cfg)
        # edge density comes from preprocess debug
        edge_list.append(float(pre_tmp.debug["glare_spec"]["edge_density"]))
        glare_stats_list.append(glare_stats_one_frame(ensure_bgr_u8(bgr), pre_cfg))

    edge_med = float(np.median(edge_list)) if edge_list else 0.0
    cfg_video, adapt_dbg = adapt_glare_params_from_stats(glare_stats_list, pre_cfg, edge_med)
    return cfg_video, adapt_dbg


# -------------------------
# pair selection (step2)
# -------------------------
def choose_adjacent_pairs(n_frames: int, k_pairs: int, rng: random.Random) -> List[Tuple[int, int]]:
    """Randomly select k adjacent pairs (i-1, i)."""
    if n_frames < 2:
        return []
    cand = list(range(1, n_frames))
    rng.shuffle(cand)
    take = cand[: min(int(k_pairs), len(cand))]
    take.sort()
    return [(i - 1, i) for i in take]


# ============================================================
# Step PRE (旧 step1)
# ============================================================
def run_step_pre(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    only_videos: Optional[List[str]] = None,
    overwrite: bool = False,
    sample_k: int = 5,
    save_first_n: int = 2,
    adapt_first_n_frames: int = 10,
    rng_seed: int = 123,
) -> None:
    """
    Step PRE:
    - per-video adapt glare on first N frames (subtitle fixed)
    - sample K frames -> save overlays/masks for first save_first_n
    - save per-video summary.json and global summary.json
    """
    ensure_dir(out_root)
    rng = random.Random(int(rng_seed))

    videos = select_videos(data_root, video_set="all", only_videos=only_videos)
    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if not frames:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        # A) adapt glare per video (subtitle fixed)
        pre_cfg_video, adapt_dbg = adapt_glare_for_video(frames, pre_cfg, adapt_first_n_frames)

        # B) sample frames
        idxs = list(range(len(frames)))
        rng.shuffle(idxs)
        pick = idxs[: min(int(sample_k), len(idxs))]
        pick.sort()

        frames_summary: List[Dict[str, Any]] = []
        saved = 0

        for idx in pick:
            fp = frames[idx]
            bgr = imread_bgr(fp)
            if bgr is None:
                continue

            pre = preprocess_frame(bgr, pre_cfg_video)
            st = compute_pre_stats(pre)

            frames_summary.append({
                "frame": os.path.basename(fp),
                "stats": st,
                "debug": pre.debug,
            })

            if saved < int(save_first_n):
                spec_overlay = overlay_mask(bgr, pre.spec_mask, (0, 0, 255), alpha=0.35)   # red
                valid_overlay = overlay_mask(bgr, pre.valid_mask, (255, 0, 0), alpha=0.35) # blue

                imwrite(os.path.join(vout, f"sample{saved+1:02d}_spec_overlay.jpg"), spec_overlay, overwrite)
                imwrite(os.path.join(vout, f"sample{saved+1:02d}_valid_overlay.jpg"), valid_overlay, overwrite)
                imwrite(os.path.join(vout, f"sample{saved+1:02d}_spec_mask.png"), pre.spec_mask, overwrite)
                imwrite(os.path.join(vout, f"sample{saved+1:02d}_valid_mask.png"), pre.valid_mask, overwrite)
                saved += 1

        summary_video = {
            "video": v,
            "n_frames_total": int(len(frames)),
            "sample_frames": [fr["frame"] for fr in frames_summary],

            "cfg_subtitle_fixed": {
                "subtitle_roi_y0_ratio": float(pre_cfg.subtitle_roi_y0_ratio),
                "sub_spec_v_min": int(pre_cfg.sub_spec_v_min),
                "sub_spec_delta_th": float(pre_cfg.sub_spec_delta_th),
            },

            "cfg_glare_final": {
                "glare_spec_v_min": int(pre_cfg_video.glare_spec_v_min),
                "glare_spec_delta_th": float(pre_cfg_video.glare_spec_delta_th),
                "spec_texture_edge_density_min": float(pre_cfg_video.spec_texture_edge_density_min),
            },

            "adapt_debug": adapt_dbg,

            "stats_mean": {
                "invalid_ratio": mean_key(frames_summary, "invalid_ratio"),
                "spec_ratio_valid": mean_key(frames_summary, "spec_ratio_valid"),
                "edge_density": mean_key(frames_summary, "edge_density"),
            },

            "frames": frames_summary,
        }

        write_json(os.path.join(vout, "summary.json"), summary_video)
        global_summary.append(summary_video)

    write_json(os.path.join(out_root, "summary.json"), global_summary)


# ============================================================
# Step MOTION (旧 step2-only)
# ============================================================
def run_step_motion(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    video_set: str = "eg",                 # "eg" or "all"
    only_videos: Optional[List[str]] = None,
    k_pairs_per_video: int = 3,
    overwrite: bool = False,
    rng_seed: int = 123,
) -> None:
    """
    Step MOTION:
    - sample adjacent pairs
    - use LoG for translation estimation
    - warp intensity for diff visualization
    - save pair overlay/diff + debug_step2.json + global summary_step2.json
    """
    ensure_dir(out_root)
    rng = random.Random(int(rng_seed))

    videos = select_videos(data_root, video_set=video_set, only_videos=only_videos)
    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if len(frames) < 2:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        pairs = choose_adjacent_pairs(len(frames), int(k_pairs_per_video), rng)
        if not pairs:
            continue

        per_video_dbg: Dict[str, Any] = {
            "video": v,
            "n_frames_total": int(len(frames)),
            "k_pairs": int(len(pairs)),
            "pairs": [],
            "pre_cfg": {
                "subtitle_mask_mode": pre_cfg.subtitle_mask_mode,
                "subtitle_roi_y0_ratio": float(pre_cfg.subtitle_roi_y0_ratio),
                "sub_spec_v_min": int(pre_cfg.sub_spec_v_min),
                "sub_spec_delta_th": float(pre_cfg.sub_spec_delta_th),
                "glare_spec_v_min": int(pre_cfg.glare_spec_v_min),
                "glare_spec_delta_th": float(pre_cfg.glare_spec_delta_th),
                "spec_enable_mode": pre_cfg.spec_enable_mode,
                "glare_fill_enable": bool(pre_cfg.glare_fill_enable),
                "glare_fill_v_offset": int(pre_cfg.glare_fill_v_offset),
                "glare_fill_dilate_ksize": int(pre_cfg.glare_fill_dilate_ksize),
            },
            "cam_cfg": {
                "roi_mode": cam_cfg.roi_mode,
                "use_soft_spec": bool(cam_cfg.use_soft_spec),
                "soft_spec_alpha": float(cam_cfg.soft_spec_alpha),
                "soft_spec_blur_sigma": float(cam_cfg.soft_spec_blur_sigma),
                "soft_spec_max_ratio": float(cam_cfg.soft_spec_max_ratio),
                "global_pc_resp_thresh": float(cam_cfg.global_pc_resp_thresh),
                "global_err_ratio_thresh": float(cam_cfg.global_err_ratio_thresh),
                "global_min_improve": float(cam_cfg.global_min_improve),
                "enable_ecc_fallback": bool(cam_cfg.enable_ecc_fallback),
            },
        }

        saved = 0
        for pi, (i0, i1) in enumerate(pairs):
            fp0, fp1 = frames[i0], frames[i1]
            bgr0 = imread_bgr(fp0)
            bgr1 = imread_bgr(fp1)
            if bgr0 is None or bgr1 is None:
                continue

            pre0 = preprocess_frame(bgr0, pre_cfg)
            pre1 = preprocess_frame(bgr1, pre_cfg)

            # ✅ 更稳：valid 用交集
            valid_use = merge_valid_intersection(pre0.valid_mask, pre1.valid_mask)

            # LoG estimation, warp intensity
            res = estimate_camera_translation(
                prev_feat=pre0.log,
                curr_feat=pre1.log,
                valid_mask=valid_use,            # hard
                prev_spec_mask=pre0.spec_mask,   # soft
                curr_spec_mask=pre1.spec_mask,   # soft
                cfg=cam_cfg,
                warp_src=pre0.intensity,         # warp intensity
            )

            dx = float(res.debug.get("final_dx", 0.0))
            dy = float(res.debug.get("final_dy", 0.0))
            method = str(res.debug.get("method", "unknown"))
            cam_moving = bool(res.debug.get("camera_moving", False))

            rec = {
                "pair_index": int(pi),
                "prev_frame": os.path.basename(fp0),
                "curr_frame": os.path.basename(fp1),
                "dx": dx,
                "dy": dy,
                "method": method,
                "camera_moving": cam_moving,
                "debug": res.debug,
            }

            if saved < int(k_pairs_per_video):
                overlay = draw_dxdy_tag(bgr1, dx, dy, method, f"pair{saved+1:02d}",
                                        extra=f"moving={int(cam_moving)}")
                diff_vis = make_diff_vis(pre1.intensity, res.prev_aligned)

                imwrite(os.path.join(vout, f"pair{saved+1:02d}_overlay_dxdy.jpg"), overlay, overwrite)
                imwrite(os.path.join(vout, f"pair{saved+1:02d}_diff.jpg"), diff_vis, overwrite)

                rec["saved_overlay"] = f"pair{saved+1:02d}_overlay_dxdy.jpg"
                rec["saved_diff"] = f"pair{saved+1:02d}_diff.jpg"
                saved += 1

            per_video_dbg["pairs"].append(rec)

        write_json(os.path.join(vout, "debug_step2.json"), per_video_dbg)

        global_summary.append({
            "video": v,
            "n_frames_total": int(len(frames)),
            "n_pairs_ran": int(len(per_video_dbg["pairs"])),
            "out_dir": vout,
            "debug_json": "debug_step2.json",
        })

    write_json(os.path.join(out_root, "summary_step2.json"), global_summary)


# ============================================================
# Step CAND (旧 step3-only)
# ============================================================
def run_step_cand(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    cand_cfg: CandidateGenConfig,
    video_set: str = "eg",
    only_videos: Optional[List[str]] = None,
    overwrite: bool = False,
) -> None:
    """
    Step CAND:
    - in-memory run Step1 + Step2 for each adjacent pair (t-1, t)
    - Step2: LoG estimation, warp intensity
    - Step3: generate candidates, save ALL overlay/mask/diff per curr frame
    - save debug_step3.json + summary_step3.json
    """
    ensure_dir(out_root)
    videos = select_videos(data_root, video_set=video_set, only_videos=only_videos)
    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if len(frames) < 2:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        gen = MotionCandidateGenerator(cand_cfg)

        per_video_dbg: Dict[str, Any] = {
            "video": v,
            "n_frames_total": int(len(frames)),
            "pairs_total": int(len(frames) - 1),
            "frames": [],
            "pre_cfg": {
                "subtitle_roi_y0_ratio": float(pre_cfg.subtitle_roi_y0_ratio),
                "spec_enable_mode": pre_cfg.spec_enable_mode,
                "glare_spec_v_min": int(pre_cfg.glare_spec_v_min),
                "glare_spec_delta_th": float(pre_cfg.glare_spec_delta_th),
            },
            "cam_cfg": {
                "roi_mode": cam_cfg.roi_mode,
                "use_soft_spec": bool(cam_cfg.use_soft_spec),
                "soft_spec_alpha": float(cam_cfg.soft_spec_alpha),
                "soft_spec_blur_sigma": float(cam_cfg.soft_spec_blur_sigma),
            },
            "cand_cfg": {
                "mad_k": float(cand_cfg.mad_k),
                "specular_weight": float(cand_cfg.specular_weight),
                "min_area": int(cand_cfg.min_area),
                "close_ksize": int(cand_cfg.close_ksize),
                "use_bridge": bool(cand_cfg.use_bridge),
                "max_boxes": int(cand_cfg.max_boxes),
            },
        }

        for i in range(1, len(frames)):
            fp0, fp1 = frames[i - 1], frames[i]
            bgr0 = imread_bgr(fp0)
            bgr1 = imread_bgr(fp1)
            if bgr0 is None or bgr1 is None:
                continue

            # Step1 (in memory)
            pre0 = preprocess_frame(bgr0, pre_cfg)
            pre1 = preprocess_frame(bgr1, pre_cfg)

            # masks merge (更稳)
            valid_use = merge_valid_intersection(pre0.valid_mask, pre1.valid_mask)
            if valid_use is None:
                valid_use = pre1.valid_mask

            spec_use = merge_spec_union_bad(pre0.spec_mask, pre1.spec_mask)
            if spec_use is None:
                spec_use = pre1.spec_mask


            # Step2 (LoG estimation, warp intensity)
            step2 = estimate_camera_translation(
                prev_feat=pre0.log,
                curr_feat=pre1.log,
                valid_mask=valid_use,
                prev_spec_mask=pre0.spec_mask,
                curr_spec_mask=pre1.spec_mask,
                cfg=cam_cfg,
                warp_src=pre0.intensity,
            )

            dx = float(step2.debug.get("final_dx", 0.0))
            dy = float(step2.debug.get("final_dy", 0.0))
            method = str(step2.debug.get("method", "unknown"))
            cam_moving = bool(step2.debug.get("camera_moving", False))

            prev_int_aligned = step2.prev_aligned
            prev_log_aligned = warp_u8(pre0.log, step2.T)

            # Step3
            r3 = generate_motion_candidates(
                curr_intensity=pre1.intensity,
                prev_intensity_aligned=prev_int_aligned,
                valid_mask=valid_use,
                spec_mask=spec_use,
                gen=gen,
                camera_moving=cam_moving,
                curr_log=pre1.log,
                prev_log_aligned=prev_log_aligned,
            )

            # save ALL for curr frame
            stem = os.path.splitext(os.path.basename(fp1))[0]
            mask_name = f"{stem}_mask.png"
            diff_name = f"{stem}_diff.jpg"
            overlay_name = f"{stem}_overlay.jpg"

            diff_vis = make_diff_vis(pre1.intensity, prev_int_aligned)
            overlay = draw_dxdy_tag(
                bgr1, dx, dy, method,
                title=f"{os.path.basename(fp0)}->{os.path.basename(fp1)}",
                extra=f"moving={int(cam_moving)} boxes={len(r3.boxes)}",
            )
            overlay = draw_boxes_on_bgr(overlay, r3.boxes)

            imwrite(os.path.join(vout, mask_name), r3.mask, overwrite)
            imwrite(os.path.join(vout, diff_name), diff_vis, overwrite)
            imwrite(os.path.join(vout, overlay_name), overlay, overwrite)

            step2_summary = {
                "dx": dx, "dy": dy, "method": method, "camera_moving": cam_moving,
                "shift_norm": float(step2.debug.get("shift_norm", 0.0)),
            }

            item = {
                "idx": int(i),
                "prev_frame": os.path.basename(fp0),
                "curr_frame": os.path.basename(fp1),
                "saved": {"mask": mask_name, "diff": diff_name, "overlay": overlay_name},
                "step2": step2_summary,
                "step3": {
                    "n_boxes": int(len(r3.boxes)),
                    "boxes": r3.boxes,
                    "debug": r3.debug,
                }
            }

            per_video_dbg["frames"].append(item)

        write_json(os.path.join(vout, "debug_step3.json"), per_video_dbg)

        global_summary.append({
            "video": v,
            "n_frames_total": int(len(frames)),
            "pairs_total": int(len(frames) - 1),
            "out_dir": vout,
            "debug_json": "debug_step3.json",
        })

    write_json(os.path.join(out_root, "summary_step3.json"), global_summary)

