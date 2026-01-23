# pipeline.py
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Iterable
import os
import json
import cv2
import numpy as np

from preprocessing import PreprocessConfig, preprocess_frame, to_grayscale, apply_subtitle_mask
from camera_motion import CamMotionConfig, estimate_camera_transform, warp_prev_to_curr, transform_translation_px
from candidate_generation import candidateGenConfig, for_pair, make_overlay


EG_VIDEOS = ["Ac4002", "Ac3004", "Ci2001", "Ci3001", "Gr3001", "Gr5009", "Su2001"]# take some examples
IMG_EXTS = (".jpg", ".png", ".jpeg")# 目前只有jpg格式的图片，可按需添加其他格式


def _is_image(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)


def list_videos(data_root: str, dataset: str, only_videos: Optional[List[str]] = None) -> List[str]:
    """
    data_root 下的一级子目录视为 video（每个 video 目录里是帧图像）。
    dataset:
      - "eg": 仅 EG_VIDEOS
      - "all": 所有子目录
    only_videos: 进一步手动指定子集（优先级最高）
    """
    all_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    if dataset == "eg":
        s = set(EG_VIDEOS)
        videos = [d for d in all_dirs if d in s]
    elif dataset == "all":
        videos = all_dirs
    else:
        raise ValueError("--dataset must be eg or all")

    if only_videos:
        ss = set(only_videos)
        videos = [v for v in videos if v in ss]
    return videos


def load_frame_order_from_json(ann_json_path: str, data_root: str) -> Dict[str, List[str]]:
    """
    可选：从 COCO-like json 的 images 字段读取 file_name（形如 VideoName/000001.jpg）
    并按 frame_id 排序，得到每个视频的帧顺序。
    """
    with open(ann_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])

    by_video: Dict[str, List[dict]] = {}
    for im in images:
        file_name = im.get("file_name", "")
        parts = file_name.replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        video, frame = parts[0], parts[-1]
        by_video.setdefault(video, []).append({"frame": frame, "frame_id": im.get("frame_id", 0)})

    order: Dict[str, List[str]] = {}
    for video, items in by_video.items():
        items.sort(key=lambda d: d["frame_id"])
        frames = [d["frame"] for d in items]
        vdir = os.path.join(data_root, video)
        if os.path.isdir(vdir):
            order[video] = frames
    return order


def select_frames(frames: List[str], strategy: str, num_clips: int, clip_len: int, every_n: int) -> List[str]:
    """
    采样策略（用于节省磁盘/内存）：
      - all: 全部
      - every: 每 every_n 帧取 1 帧
      - clips: 均匀抽 num_clips 段，每段 clip_len 帧（这个可能不需要）
    """
    n = len(frames)
    if n == 0:
        return []

    strategy = strategy.lower()
    if strategy == "all":
        return frames

    if strategy == "every":
        every_n = max(1, int(every_n))
        return [frames[i] for i in range(0, n, every_n)]

    # default: clips
    clip_len = max(2, int(clip_len))
    num_clips = max(1, int(num_clips))
    if n <= clip_len:
        return frames

    max_start = n - clip_len
    if num_clips == 1:
        starts = [max_start // 2]
    else:
        starts = [int(round(i * max_start / (num_clips - 1))) for i in range(num_clips)]

    picked: List[str] = []
    seen = set()
    for st in starts:
        for i in range(st, min(st + clip_len, n)):
            if i not in seen:
                picked.append(frames[i])
                seen.add(i)
    return picked


def build_frames_by_video(
    data_root: str,
    dataset: str,
    ann_json: Optional[str],
    only_videos: Optional[List[str]],
    sample_strategy: str,
    num_clips: int,
    clip_len: int,
    every_n: int,
) -> Dict[str, List[str]]:
    videos = list_videos(data_root, dataset, only_videos)

    order = load_frame_order_from_json(ann_json, data_root) if ann_json else {}

    out: Dict[str, List[str]] = {}
    for v in videos:
        vdir = os.path.join(data_root, v)
        frames = order.get(v)
        if frames is None:
            frames = sorted([f for f in os.listdir(vdir) if _is_image(f)])
        frames = select_frames(frames, sample_strategy, num_clips, clip_len, every_n)
        out[v] = frames
    return out


def run_preprocessing(data_root: str, out_pre_root: str, pre_cfg: PreprocessConfig,
                    frames_by_video: Dict[str, List[str]], overwrite: bool) -> None:
    """Step1：保存预处理后的灰度帧。"""
    os.makedirs(out_pre_root, exist_ok=True)

    for video, frames in frames_by_video.items():
        in_dir = os.path.join(data_root, video)
        out_dir = os.path.join(out_pre_root, video)
        os.makedirs(out_dir, exist_ok=True)

        for fn in frames:
            outp = os.path.join(out_dir, fn)
            if (not overwrite) and os.path.exists(outp):
                continue
            bgr = cv2.imread(os.path.join(in_dir, fn))
            if bgr is None:
                continue
            pre = preprocess_frame(bgr, pre_cfg)
            cv2.imwrite(outp, pre)


def run_camera_motion(data_root: str, pre_root: str, aligned_root: str,
                          pre_cfg: PreprocessConfig, cam_cfg: CamMotionConfig,
                          frames_by_video: Dict[str, List[str]], overwrite: bool) -> None:
    """
    Step2：读取 Step1 结果并保存 prev_aligned（对齐后的上一帧）+ transforms json（用于诊断门控/匹配质量）。
    """
    os.makedirs(aligned_root, exist_ok=True)

    for video, frames in frames_by_video.items():
        raw_dir = os.path.join(data_root, video)
        pre_dir = os.path.join(pre_root, video)
        out_dir = os.path.join(aligned_root, video)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(pre_dir):
            raise FileNotFoundError(f"Missing Step1 output: {pre_dir}")

        transforms: Dict[str, dict] = {}
        prev_gray_motion = None
        prev_pre = None
        prev_name = None

        for i, fn in enumerate(frames):
            raw = cv2.imread(os.path.join(raw_dir, fn))
            if raw is None:
                continue

            # motion 用 raw 灰度（但也应用同样的字幕 mask，减少字幕特征干扰）
            curr_gray_motion = to_grayscale(raw)
            curr_gray_motion = apply_subtitle_mask(
                curr_gray_motion,
                pre_cfg.subtitle_mask_mode,
                pre_cfg.subtitle_mask_ratio,
                pre_cfg.auto_edge_frac_thresh,
                pre_cfg.auto_canny1,
                pre_cfg.auto_canny2
            )

            curr_pre = cv2.imread(os.path.join(pre_dir, fn), cv2.IMREAD_GRAYSCALE)
            if curr_pre is None:
                raise FileNotFoundError(f"Missing preprocessed frame: {os.path.join(pre_dir, fn)}")

            outp = os.path.join(out_dir, fn)
            if (not overwrite) and os.path.exists(outp):
                prev_gray_motion = curr_gray_motion
                prev_pre = curr_pre
                prev_name = fn
                continue

            if i == 0 or prev_gray_motion is None or prev_pre is None:
                cv2.imwrite(outp, curr_pre)
                transforms[fn] = {"M": np.eye(3, dtype=np.float32).tolist(), "inliers": 0, "trans_px": 0.0, "prev": None}
            else:
                M, inliers = estimate_camera_transform(prev_gray_motion, curr_gray_motion, cam_cfg)
                trans_px = transform_translation_px(M)
                prevA = warp_prev_to_curr(prev_pre, M, curr_pre.shape[:2], cam_cfg)
                cv2.imwrite(outp, prevA)
                transforms[fn] = {"M": M.tolist(), "inliers": int(inliers), "trans_px": float(trans_px), "prev": prev_name}

            prev_gray_motion = curr_gray_motion
            prev_pre = curr_pre
            prev_name = fn

        tf_path = os.path.join(out_dir, f"transforms_{video}.json")
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump({
                "video": video,
                "pre_cfg": asdict(pre_cfg),
                "cam_cfg": asdict(cam_cfg),
                "frames": transforms
            }, f, ensure_ascii=False, indent=2)


def run_candidate_generation_streaming(
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    cam_cfg: CamMotionConfig,
    candidate_gen_cfg: candidateGenConfig,
    frames_by_video: Dict[str, List[str]],
    overwrite: bool
) -> None:
    """
    Step3（单独执行）= Step1->Step2->Step3 串联，但默认不落盘 Step1/2，
    只保存 Step3 输出（diff/mask/overlay + candidates json），便于判断需要调哪一步。
    """
    os.makedirs(out_root, exist_ok=True)
    diff_root = os.path.join(out_root, "diff")
    mask_root = os.path.join(out_root, "mask")
    overlay_root = os.path.join(out_root, "overlay")
    cand_root = os.path.join(out_root, "candidates")

    if candidate_gen_cfg.save_diff:
        os.makedirs(diff_root, exist_ok=True)
    if candidate_gen_cfg.save_mask:
        os.makedirs(mask_root, exist_ok=True)
    if candidate_gen_cfg.save_overlay:
        os.makedirs(overlay_root, exist_ok=True)
    os.makedirs(cand_root, exist_ok=True)

    for video, frames in frames_by_video.items():
        raw_dir = os.path.join(data_root, video)
        if not frames:
            continue

        if candidate_gen_cfg.save_diff:
            os.makedirs(os.path.join(diff_root, video), exist_ok=True)
        if candidate_gen_cfg.save_mask:
            os.makedirs(os.path.join(mask_root, video), exist_ok=True)
        if candidate_gen_cfg.save_overlay:
            os.makedirs(os.path.join(overlay_root, video), exist_ok=True)

        records = {
            "video": video,
            "pre_cfg": asdict(pre_cfg),
            "cam_cfg": asdict(cam_cfg),
            "candidate_gen_cfg": asdict(candidate_gen_cfg),
            "frames": {}
        }

        prev_gray_motion = None
        prev_pre = None
        prev_name = None

        for i, fn in enumerate(frames):
            bgr = cv2.imread(os.path.join(raw_dir, fn))
            if bgr is None:
                continue

            curr_gray_motion = to_grayscale(bgr)
            curr_gray_motion = apply_subtitle_mask(
                curr_gray_motion,
                pre_cfg.subtitle_mask_mode,
                pre_cfg.subtitle_mask_ratio,
                pre_cfg.auto_edge_frac_thresh,
                pre_cfg.auto_canny1,
                pre_cfg.auto_canny2
            )
            curr_pre = preprocess_frame(bgr, pre_cfg)

            if i == 0 or prev_gray_motion is None or prev_pre is None:
                M = np.eye(3, dtype=np.float32)
                inliers = 0
                trans_px = 0.0
                prevA = curr_pre
            else:
                M, inliers = estimate_camera_transform(prev_gray_motion, curr_gray_motion, cam_cfg)
                trans_px = transform_translation_px(M)
                prevA = warp_prev_to_curr(prev_pre, M, curr_pre.shape[:2], cam_cfg)

            diff, mask2, bboxes, T = for_pair(curr_pre, prevA, candidate_gen_cfg)

            records["frames"][fn] = {
                "threshold": int(T),
                "num_candidates": int(len(bboxes)),
                "bboxes": bboxes,
                "cam_inliers": int(inliers),
                "cam_trans_px": float(trans_px),
                "prev_frame": prev_name
            }

            if candidate_gen_cfg.save_diff:
                outp = os.path.join(diff_root, video, fn)
                if overwrite or (not os.path.exists(outp)):
                    cv2.imwrite(outp, diff)

            if candidate_gen_cfg.save_mask:
                outp = os.path.join(mask_root, video, fn)
                if overwrite or (not os.path.exists(outp)):
                    cv2.imwrite(outp, mask2)

            if candidate_gen_cfg.save_overlay:
                outp = os.path.join(overlay_root, video, fn)
                if overwrite or (not os.path.exists(outp)):
                    vis = make_overlay(curr_pre, bboxes, max_draw=candidate_gen_cfg.max_candidates_draw)
                    cv2.imwrite(outp, vis)

            prev_gray_motion = curr_gray_motion
            prev_pre = curr_pre
            prev_name = fn

        out_json = os.path.join(cand_root, f"{video}_candidates.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)