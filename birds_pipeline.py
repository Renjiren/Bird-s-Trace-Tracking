# birds_pipeline.py
from __future__ import annotations

import os
import json
import random
import time
from time import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from preprocessing import PreprocessConfig, preprocess_frame
from camera_motion_compensation import CamMotionConfig, estimate_camera_translation, warp_u8
from candidate_generation import CandidateGenConfig, MotionCandidateGenerator, generate_motion_candidates
from step4_refine import RefineConfig, step4_refine
from step5_tracker import Tracker


IMG_EXTS = (".jpg", ".jpeg", ".png")

EG_VIDEOS = [
    "Ac4002", "Ac4003", "An3004", "An3013",
    "An6011", "Ci2001", "Ci3001", "Pa1003",
    "Gr5009", "Su2001", "Su2002", "Su2005"
]


# ---------------- IO helpers ----------------
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


# ---------------- visualization ----------------
def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    base = bgr.copy()
    paint = base.copy()
    paint[mask == 0] = color_bgr
    return cv2.addWeighted(paint, float(alpha), base, 1.0 - float(alpha), 0.0)


def draw_boxes(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], max_draw: int = 80) -> np.ndarray:
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


def merge_valid_intersection(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None:
        return b
    if b is None:
        return a
    if a.shape != b.shape:
        return b
    return cv2.bitwise_and(a, b)


def merge_spec_union_bad(prev_spec: Optional[np.ndarray], curr_spec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    # union of BAD(0) for 0/255 masks == bitwise_and
    if prev_spec is None:
        return curr_spec
    if curr_spec is None:
        return prev_spec
    if prev_spec.shape != curr_spec.shape:
        return curr_spec
    return cv2.bitwise_and(prev_spec, curr_spec)


# ============================================================
# Step1: PRE
# ============================================================
def run_step_pre(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    video_set: str,
    only_videos: Optional[List[str]],
    overwrite: bool,
    rng_seed: int,
    sample_k: int = 5,
    save_first_n: int = 2,
) -> None:
    ensure_dir(out_root)
    rng = random.Random(int(rng_seed))

    videos = select_videos(data_root, video_set, only_videos)
    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if not frames:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        idxs = list(range(len(frames)))
        rng.shuffle(idxs)
        pick = idxs[: min(int(sample_k), len(idxs))]
        pick.sort()

        saved = 0
        per_video_frames: List[Dict[str, Any]] = []

        for idx in pick:
            fp = frames[idx]
            bgr = imread_bgr(fp)
            if bgr is None:
                continue

            pre = preprocess_frame(bgr, pre_cfg)
            per_video_frames.append({
                "frame": os.path.basename(fp),
                "stats": {
                    "valid_ratio": float(np.mean(pre.valid_mask > 0)),
                    "spec_ratio_valid": float(np.mean((pre.spec_mask == 0)[pre.valid_mask > 0])) if np.any(pre.valid_mask > 0) else 0.0,
                    "edge_density": float(np.mean((pre.log >= pre_cfg.log_edge_th)[pre.valid_mask > 0])) if np.any(pre.valid_mask > 0) else 0.0,
                },
            })

            if saved < int(save_first_n):
                imwrite(os.path.join(vout, f"sample{saved+1:02d}_valid_overlay.jpg"),
                        overlay_mask(bgr, pre.valid_mask, (255, 0, 0), 0.35), overwrite)
                imwrite(os.path.join(vout, f"sample{saved+1:02d}_spec_overlay.jpg"),
                        overlay_mask(bgr, pre.spec_mask, (0, 0, 255), 0.35), overwrite)
                saved += 1

        summary_video = {
            "video": v,
            "n_frames_total": int(len(frames)),
            "pre_cfg": {
                "subtitle_mask_mode": pre_cfg.subtitle_mask_mode,
                "smooth_mode": pre_cfg.smooth_mode,
                "spec_enable_mode": pre_cfg.spec_enable_mode,
                "subtitle_roi_y0_ratio": float(pre_cfg.subtitle_roi_y0_ratio),
            },
            "frames": per_video_frames,
        }
        write_json(os.path.join(vout, "summary.json"), summary_video)
        global_summary.append(summary_video)

    write_json(os.path.join(out_root, "summary.json"), global_summary)


# ============================================================
# Step2: MOTION (sample pairs)
# ============================================================
def run_step_motion(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    video_set: str,
    only_videos: Optional[List[str]],
    overwrite: bool,
    rng_seed: int,
    k_pairs_per_video: int = 3,
) -> None:
    ensure_dir(out_root)
    rng = random.Random(int(rng_seed))

    videos = select_videos(data_root, video_set, only_videos)
    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if len(frames) < 2:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        # random adjacent pairs
        cand = list(range(1, len(frames)))
        rng.shuffle(cand)
        take = sorted(cand[: min(int(k_pairs_per_video), len(cand))])
        pairs = [(i - 1, i) for i in take]

        per_video_dbg: Dict[str, Any] = {"video": v, "pairs": []}

        for pi, (i0, i1) in enumerate(pairs):
            fp0, fp1 = frames[i0], frames[i1]
            bgr0 = imread_bgr(fp0)
            bgr1 = imread_bgr(fp1)
            if bgr0 is None or bgr1 is None:
                continue

            pre0 = preprocess_frame(bgr0, pre_cfg)
            pre1 = preprocess_frame(bgr1, pre_cfg)

            valid_use = merge_valid_intersection(pre0.valid_mask, pre1.valid_mask) or pre1.valid_mask

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

            diff_vis = make_diff_vis(pre1.intensity, step2.prev_aligned)
            overlay = bgr1.copy()
            cv2.putText(overlay, f"dx={dx:+.2f} dy={dy:+.2f} {method} moving={int(cam_moving)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            imwrite(os.path.join(vout, f"pair{pi+1:02d}_diff.jpg"), diff_vis, overwrite)
            imwrite(os.path.join(vout, f"pair{pi+1:02d}_overlay.jpg"), overlay, overwrite)

            per_video_dbg["pairs"].append({
                "pair_index": int(pi),
                "prev_frame": os.path.basename(fp0),
                "curr_frame": os.path.basename(fp1),
                "dx": dx,
                "dy": dy,
                "method": method,
                "camera_moving": cam_moving,
            })

        write_json(os.path.join(vout, "debug_step2.json"), per_video_dbg)
        global_summary.append({"video": v, "debug_json": "debug_step2.json", "out_dir": vout})

    write_json(os.path.join(out_root, "summary_step2.json"), global_summary)


# ============================================================
# Step3: CAND (all adjacent pairs)
# ============================================================
def run_step_cand(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    cand_cfg: CandidateGenConfig,
    video_set: str,
    only_videos: Optional[List[str]],
    overwrite: bool,
) -> None:
    ensure_dir(out_root)

    videos = select_videos(data_root, video_set, only_videos)
    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if len(frames) < 2:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        gen = MotionCandidateGenerator(cand_cfg)
        per_video_dbg: Dict[str, Any] = {"video": v, "frames": []}

        for i in range(1, len(frames)):
            fp0, fp1 = frames[i - 1], frames[i]
            bgr0 = imread_bgr(fp0)
            bgr1 = imread_bgr(fp1)
            if bgr0 is None or bgr1 is None:
                continue

            pre0 = preprocess_frame(bgr0, pre_cfg)
            pre1 = preprocess_frame(bgr1, pre_cfg)

            valid_use = merge_valid_intersection(pre0.valid_mask, pre1.valid_mask) or pre1.valid_mask
            spec_use = merge_spec_union_bad(pre0.spec_mask, pre1.spec_mask) or pre1.spec_mask

            step2 = estimate_camera_translation(
                prev_feat=pre0.log,
                curr_feat=pre1.log,
                valid_mask=valid_use,
                prev_spec_mask=pre0.spec_mask,
                curr_spec_mask=pre1.spec_mask,
                cfg=cam_cfg,
                warp_src=pre0.intensity,
            )

            prev_int_aligned = step2.prev_aligned
            prev_log_aligned = warp_u8(pre0.log, step2.T)
            cam_moving = bool(step2.debug.get("camera_moving", False))

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

            stem = os.path.splitext(os.path.basename(fp1))[0]
            mask_name = f"{stem}_mask.png"
            overlay_name = f"{stem}_overlay.jpg"

            imwrite(os.path.join(vout, mask_name), r3.mask, overwrite)
            imwrite(os.path.join(vout, overlay_name), draw_boxes(bgr1, r3.boxes), overwrite)

            per_video_dbg["frames"].append({
                "idx": int(i),
                "prev_frame": os.path.basename(fp0),
                "curr_frame": os.path.basename(fp1),
                "n_boxes": int(len(r3.boxes)),
                "saved": {"mask": mask_name, "overlay": overlay_name},
                "step2": {
                    "dx": float(step2.debug.get("final_dx", 0.0)),
                    "dy": float(step2.debug.get("final_dy", 0.0)),
                    "method": str(step2.debug.get("method", "unknown")),
                    "camera_moving": cam_moving,
                },
            })

        write_json(os.path.join(vout, "debug_step3.json"), per_video_dbg)
        global_summary.append({"video": v, "debug_json": "debug_step3.json", "out_dir": vout})

    write_json(os.path.join(out_root, "summary_step3.json"), global_summary)


# ============================================================
# Step REFINE 
# ============================================================
def run_step_refine(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    cand_cfg: CandidateGenConfig,
    refine_cfg: RefineConfig,
    video_set: str = "eg",
    only_videos: Optional[List[str]] = None,
    overwrite: bool = False,
) -> None:
    """
    Step1 + Step2 + Step3 + Step4 in ONE PASS
    完全模仿 run_step_cand，只在末尾接 refine
    """

    ensure_dir(out_root)
    videos = select_videos(data_root, video_set=video_set, only_videos=only_videos)

    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if len(frames) < 2:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        gen = MotionCandidateGenerator(cand_cfg)

        per_video_dbg = {
            "video": v,
            "frames": [],
        }

        for i in range(1, len(frames)):
            fp0, fp1 = frames[i - 1], frames[i]
            bgr0 = imread_bgr(fp0)
            bgr1 = imread_bgr(fp1)
            if bgr0 is None or bgr1 is None:
                continue

            # ---------- Step1 ----------
            pre0 = preprocess_frame(bgr0, pre_cfg)
            pre1 = preprocess_frame(bgr1, pre_cfg)

            valid_use = merge_valid_intersection(pre0.valid_mask, pre1.valid_mask)
            spec_use = merge_spec_union_bad(pre0.spec_mask, pre1.spec_mask)

            # ---------- Step2 ----------
            step2 = estimate_camera_translation(
                prev_feat=pre0.log,
                curr_feat=pre1.log,
                valid_mask=valid_use,
                prev_spec_mask=pre0.spec_mask,
                curr_spec_mask=pre1.spec_mask,
                cfg=cam_cfg,
                warp_src=pre0.intensity,
            )

            prev_int_aligned = step2.prev_aligned
            prev_log_aligned = warp_u8(pre0.log, step2.T)

            # ---------- Step3 ----------
            r3 = generate_motion_candidates(
                curr_intensity=pre1.intensity,
                prev_intensity_aligned=prev_int_aligned,
                valid_mask=valid_use,
                spec_mask=spec_use,
                gen=gen,
                camera_moving=bool(step2.debug.get("camera_moving", False)),
                curr_log=pre1.log,
                prev_log_aligned=prev_log_aligned,
            )

            # ---------- Step4 ----------
            spec_mask_curr = pre1.spec_mask

            boxes_refined = step4_refine(
                bgr=bgr1,
                mask_fg=r3.mask,
                boxes_raw=r3.boxes,
                spec_mask=spec_mask_curr,
                cfg=refine_cfg,
            )

            # ---------- Save ----------
            stem = os.path.splitext(os.path.basename(fp1))[0]
            overlay = draw_boxes_on_bgr(bgr1, boxes_refined)
            out_name = f"{stem}_overlay_refine.jpg"
            imwrite(os.path.join(vout, out_name), overlay, overwrite)

            per_video_dbg["frames"].append({
                "idx": i,
                "curr_frame": os.path.basename(fp1),
                "n_step3": len(r3.boxes),
                "n_step4": len(boxes_refined),
                "boxes_refined": boxes_refined,
                "saved": out_name,
            })

        write_json(os.path.join(vout, "debug_step4.json"), per_video_dbg)



# ============================================================
# Step TRACKER (step 5)
# ============================================================
def run_step_track(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    cand_cfg: CandidateGenConfig,
    refine_cfg: RefineConfig,
    video_set: str = "eg",
    only_videos: Optional[List[str]] = None,
    overwrite: bool = False,
) -> None:
    """
    Step1–4 + Step5 tracking
    在 run_step_refine 基础上，加 tracker
    """
    ensure_dir(out_root)
    videos = select_videos(data_root, video_set=video_set, only_videos=only_videos)

    for v in videos:
        t_v_start = time.perf_counter()
        vdir = os.path.join(data_root, v)
        frames = list_frames(vdir)
        if len(frames) < 2:
            continue

        vout = os.path.join(out_root, v)
        ensure_dir(vout)

        gen = MotionCandidateGenerator(cand_cfg)
        tracker = Tracker(iou_thr=0.3, max_age=5, min_hits=2)

        per_video_dbg = {
            "video": v,
            "frames": [],
        }

        txt_path = os.path.join(out_root, f"{v}.txt")
        f_txt = open(txt_path, "w", encoding="utf-8")

        for i in range(1, len(frames)):
            fp0, fp1 = frames[i - 1], frames[i]
            bgr0 = imread_bgr(fp0)
            bgr1 = imread_bgr(fp1)
            if bgr0 is None or bgr1 is None:
                continue

            prev_gray = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)

            # ---------- Step1 ----------
            pre0 = preprocess_frame(bgr0, pre_cfg)
            pre1 = preprocess_frame(bgr1, pre_cfg)

            valid_use = merge_valid_intersection(pre0.valid_mask, pre1.valid_mask)
            spec_use = merge_spec_union_bad(pre0.spec_mask, pre1.spec_mask)

            # ---------- Step2 ----------
            step2 = estimate_camera_translation(
                prev_feat=pre0.log,
                curr_feat=pre1.log,
                valid_mask=valid_use,
                prev_spec_mask=pre0.spec_mask,
                curr_spec_mask=pre1.spec_mask,
                cfg=cam_cfg,
                warp_src=pre0.intensity,
            )

            prev_int_aligned = step2.prev_aligned
            prev_log_aligned = warp_u8(pre0.log, step2.T)

            # ---------- Step3 ----------
            r3 = generate_motion_candidates(
                curr_intensity=pre1.intensity,
                prev_intensity_aligned=prev_int_aligned,
                valid_mask=valid_use,
                spec_mask=spec_use,
                gen=gen,
                camera_moving=bool(step2.debug.get("camera_moving", False)),
                curr_log=pre1.log,
                prev_log_aligned=prev_log_aligned,
            )

            # ---------- Step4 ----------
            boxes_refined = step4_refine(
                bgr=bgr1,
                mask_fg=r3.mask,
                boxes_raw=r3.boxes,
                spec_mask=pre1.spec_mask,
                cfg=refine_cfg,
            )

            # ---------- Step5 ----------
            tracks = tracker.step(boxes_refined, prev_gray, curr_gray)
            frame_id = i  # 你这里 i 从 1 开始，正好当 frame_id

            for tid, (x, y, w, h) in tracks:
                # MOT challenge 常见格式：frame, id, x, y, w, h, score, -1, -1, -1
                f_txt.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n")

            # ---------- Save ----------
            overlay = bgr1.copy()
            for tid, box in tracks:
                x, y, w, h = box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    overlay, f"ID {tid}",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

            stem = os.path.splitext(os.path.basename(fp1))[0]
            out_name = f"{stem}_overlay_track.jpg"
            imwrite(os.path.join(vout, out_name), overlay, overwrite)

            per_video_dbg["frames"].append({
                "idx": i,
                "curr_frame": os.path.basename(fp1),
                "n_det": len(boxes_refined),
                "n_tracks": len(tracks),
                "tracks": tracks,
                "saved": out_name,
            })

        f_txt.close()
        t_v = time.perf_counter() - t_v_start
        print(f"[Step5] Video {v}: {t_v:.2f}s")

        write_json(os.path.join(vout, "debug_step5.json"), per_video_dbg)