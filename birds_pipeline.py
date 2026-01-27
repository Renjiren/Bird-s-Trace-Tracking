# birds_pipeline.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Iterable, Literal

import cv2
import numpy as np

from preprocessing import PreprocessConfig, preprocess_frame
from camera_motion_compensation import CamMotionConfig, estimate_camera_translation
from candidate_generation import (
    CandidateGenConfig,
    MotionCandidateGenerator,
    generate_motion_candidates,
    draw_overlay,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")

EG_VIDEOS = [
    "Ac4002", "An3004", "An3013",
    "An6012", "Ci2001", "Ci3001",
    "Pa1003", "Su2001", "Su2002",
    "Su2005"
]

Stage = Literal["pre", "cam", "cand"]


@dataclass(frozen=True)
class FrameItem:
    video: str
    frame_id: int
    rel_file: str
    abs_path: str


@dataclass(frozen=True)
class SaveConfig:
    """
    只控制“图片输出”的保存，不影响 results.jsonl（始终写，方便你看每一步 debug）。
    """
    save_images: bool = True
    save_every: int = 1
    overwrite: bool = False


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_image(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _infer_frame_id_from_name(rel_file: str) -> int:
    base = os.path.basename(rel_file)
    stem, _ = os.path.splitext(base)
    try:
        return int(stem)
    except Exception:
        return -1


def load_videos_from_manifest(manifest_path: str, data_root: str) -> Dict[str, List[FrameItem]]:
    """
    读取 COCO-like json：{"images":[{file_name,frame_id,...}, ...]}
    并按 video(=file_name 第一段) 分组、按 frame_id 排序。
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    by_video: Dict[str, List[FrameItem]] = {}

    for im in images:
        rel = im.get("file_name", "")
        if not rel:
            continue
        video = rel.split("/")[0]
        frame_id = int(im.get("frame_id", _infer_frame_id_from_name(rel)))
        abs_path = os.path.join(data_root, rel)
        by_video.setdefault(video, []).append(
            FrameItem(video=video, frame_id=frame_id, rel_file=rel, abs_path=abs_path)
        )

    for v in list(by_video.keys()):
        by_video[v].sort(key=lambda x: x.frame_id)
        if not by_video[v]:
            by_video.pop(v, None)

    return by_video


def load_videos_from_folders(data_root: str) -> Dict[str, List[FrameItem]]:
    """
    旧方式：扫描 data_root 下的文件夹作为视频名，每个文件夹内按文件名排序。
    """
    by_video: Dict[str, List[FrameItem]] = {}
    for v in sorted(os.listdir(data_root)):
        vdir = os.path.join(data_root, v)
        if not os.path.isdir(vdir):
            continue
        frames = [fn for fn in os.listdir(vdir) if _is_image(fn)]
        frames.sort()
        items: List[FrameItem] = []
        for fn in frames:
            rel = f"{v}/{fn}"
            items.append(
                FrameItem(
                    video=v,
                    frame_id=_infer_frame_id_from_name(rel),
                    rel_file=rel,
                    abs_path=os.path.join(data_root, rel),
                )
            )
        if items:
            items.sort(key=lambda x: x.frame_id)
            by_video[v] = items
    return by_video


def _filter_videos(by_video: Dict[str, List[FrameItem]], only_videos: Optional[List[str]]) -> Dict[str, List[FrameItem]]:
    if not only_videos:
        return by_video
    ss = set(only_videos)
    return {v: by_video[v] for v in by_video.keys() if v in ss}


def iter_frames(frames: List[FrameItem], every_n: int) -> Iterable[FrameItem]:
    if every_n <= 1:
        yield from frames
    else:
        yield from frames[::every_n]


def _maybe_save_image(path: str, img: np.ndarray, overwrite: bool) -> None:
    if (not overwrite) and os.path.exists(path):
        return
    cv2.imwrite(path, img)


def _warp_by_T(src_u8: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
    if T is None:
        return src_u8
    H, W = src_u8.shape[:2]
    return cv2.warpAffine(src_u8, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _target_stage_from_arg(stage_arg: str) -> Stage:
    # main.py 会保证 stage_arg in {"pre","cam","cand","all"}
    return "cand" if stage_arg == "all" else stage_arg  # type: ignore


def run(
    *,
    data_root: str,
    out_root: str,
    stage_arg: str,  # "pre" | "cam" | "cand" | "all"
    pre_cfg: PreprocessConfig,
    cam_cfg: Optional[CamMotionConfig],
    cand_cfg: Optional[CandidateGenConfig],
    manifest_path: Optional[str] = None,
    only_videos: Optional[List[str]] = None,
    every_n: int = 1,
    save: SaveConfig = SaveConfig(),
    ablate_no_cam_motion: bool = False,
) -> None:
    """
    统一入口：按 target stage 运行：
      - pre : 只跑 Step1
      - cam : 跑 Step1 + Step2
      - cand/all : 跑 Step1 + Step2 + Step3

    保存策略（满足你的要求）：
      - 只保存“当前 stage 的输出图片”
        pre  -> 保存 valid_mask/spec_mask
        cam  -> 保存 prev_aligned（对齐后的 prev_intensity）
        cand -> 保存 mask + overlay
      - 中间步骤的图片不保存
      - results.jsonl（每视频一个）始终写，用于调参/排错
    """
    _ensure_dir(out_root)
    target_stage: Stage = _target_stage_from_arg(stage_arg)

    # 依赖关系：cand 依赖 cam，cam 依赖 pre
    use_pre = True
    use_cam = target_stage in ("cam", "cand")
    use_cand = target_stage == "cand"

    if use_cam and cam_cfg is None:
        raise ValueError("cam stage requested but cam_cfg is None")
    if use_cand and cand_cfg is None:
        raise ValueError("cand stage requested but cand_cfg is None")

    # 加载视频索引
    if manifest_path:
        by_video = load_videos_from_manifest(manifest_path, data_root)
        source = {"type": "manifest", "manifest": manifest_path}
    else:
        by_video = load_videos_from_folders(data_root)
        source = {"type": "folders", "data_root": data_root}

    by_video = _filter_videos(by_video, only_videos)

    # meta
    meta = {
        "source": source,
        "stage": stage_arg,
        "target_stage": target_stage,
        "every_n": int(every_n),
        "ablate_no_cam_motion": bool(ablate_no_cam_motion),
        "pre_cfg": asdict(pre_cfg),
        "cam_cfg": asdict(cam_cfg) if cam_cfg else None,
        "cand_cfg": asdict(cand_cfg) if cand_cfg else None,
        "save": asdict(save),
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

    # per-video
    for v in sorted(by_video.keys()):
        frames = by_video[v]
        if not frames:
            continue

        vout = os.path.join(out_root, v)
        _ensure_dir(vout)

        # results.jsonl
        results_path = os.path.join(vout, "results.jsonl")
        rf = open(results_path, "w", encoding="utf-8")

        # stage-specific output dirs (only create when needed)
        pre_valid_dir = pre_spec_dir = None
        cam_align_dir = None
        cand_mask_dir = cand_overlay_dir = None

        if save.save_images:
            if target_stage == "pre":
                pre_valid_dir = os.path.join(vout, "pre", "valid_mask")
                pre_spec_dir = os.path.join(vout, "pre", "spec_mask")
                _ensure_dir(pre_valid_dir)
                _ensure_dir(pre_spec_dir)

            elif target_stage == "cam":
                cam_align_dir = os.path.join(vout, "cam", "prev_aligned")
                _ensure_dir(cam_align_dir)

            elif target_stage == "cand":
                cand_mask_dir = os.path.join(vout, "cand", "mask")
                cand_overlay_dir = os.path.join(vout, "cand", "overlay")
                _ensure_dir(cand_mask_dir)
                _ensure_dir(cand_overlay_dir)

        # per-video state
        gen = MotionCandidateGenerator(cand_cfg) if use_cand else None

        prev_pre = None  # previous PreprocessResult
        prev_name = None

        for idx, item in enumerate(iter_frames(frames, every_n)):
            bgr = cv2.imread(item.abs_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            fn = os.path.basename(item.rel_file)

            # ---------- Step1 ----------
            pre = preprocess_frame(bgr, pre_cfg)  # always run
            # convenience
            curr_intensity = pre.intensity
            curr_log = pre.log
            curr_valid = pre.valid_mask
            curr_spec = pre.spec_mask

            # record base
            record: Dict[str, Any] = {
                "video": v,
                "frame": fn,
                "rel_file": item.rel_file,
                "frame_id": int(item.frame_id),
                "prev": prev_name,
                "step1": pre.debug,  # 小而关键：subtitle/spec 比例、log 配置等
                "step2": None,
                "step3": None,
                "boxes": [],
            }

            # stage=pre: save ONLY step1 masks
            if target_stage == "pre" and save.save_images and (idx % max(1, save.save_every) == 0):
                _maybe_save_image(os.path.join(pre_valid_dir, fn), curr_valid, save.overwrite)  # type: ignore
                _maybe_save_image(os.path.join(pre_spec_dir, fn), curr_spec, save.overwrite)    # type: ignore

            # first frame handling
            if prev_pre is None:
                record["step2"] = {"method": "first_frame", "camera_moving": False}
                record["step3"] = {"reason": "first_frame"}
                rf.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                prev_pre, prev_name = pre, fn
                continue

            # combined valid for t-1 and t (safer than using only current valid)
            valid = cv2.bitwise_and(prev_pre.valid_mask, curr_valid)

            # ---------- Step2 ----------
            # 只要 target_stage >= cam 就需要估计运动（cand 也依赖）
            prev_aligned_intensity = prev_pre.intensity
            prev_aligned_log = prev_pre.log
            T = None
            camera_moving = False

            if use_cam:
                if ablate_no_cam_motion:
                    record["step2"] = {"method": "identity(ablate)", "camera_moving": False}
                else:
                    # 用 LoG 估计平移，warp_src = prev_intensity 输出 prev_aligned_intensity
                    res = estimate_camera_translation(
                        prev_feat=prev_pre.log,
                        curr_feat=curr_log,
                        valid_mask=valid,
                        prev_spec_mask=prev_pre.spec_mask,
                        curr_spec_mask=curr_spec,
                        cfg=cam_cfg,  # type: ignore
                        warp_src=prev_pre.intensity,
                    )

                    T = res.T
                    camera_moving = bool(res.camera_moving)
                    prev_aligned_intensity = res.prev_aligned
                    prev_aligned_log = _warp_by_T(prev_pre.log, T)

                    # 保存 step2 debug（精简 + full）
                    record["step2"] = {
                        "method": res.debug.get("method", "unknown"),
                        "dx": res.debug.get("final_dx", 0.0),
                        "dy": res.debug.get("final_dy", 0.0),
                        "shift_norm": res.debug.get("shift_norm", 0.0),
                        "camera_moving": bool(camera_moving),
                        "debug_full": res.debug,
                    }

                # stage=cam: save ONLY step2 output image
                if target_stage == "cam" and save.save_images and (idx % max(1, save.save_every) == 0):
                    _maybe_save_image(os.path.join(cam_align_dir, fn), prev_aligned_intensity, save.overwrite)  # type: ignore

            else:
                record["step2"] = {"reason": "skipped"}

            # ---------- Step3 ----------
            if use_cand and gen is not None:
                step3 = generate_motion_candidates(
                    curr_intensity=curr_intensity,
                    prev_intensity_aligned=prev_aligned_intensity,
                    valid_mask=valid,
                    spec_mask=curr_spec,
                    gen=gen,
                    camera_moving=camera_moving,
                    curr_log=curr_log,
                    prev_log_aligned=prev_aligned_log,
                )
                record["boxes"] = step3.boxes
                record["step3"] = step3.debug

                # stage=cand: save ONLY step3 outputs
                if target_stage == "cand" and save.save_images and (idx % max(1, save.save_every) == 0):
                    _maybe_save_image(os.path.join(cand_mask_dir, fn), step3.mask, save.overwrite)  # type: ignore

                    vis = draw_overlay(curr_intensity, step3.boxes)
                    if isinstance(vis, tuple):
                        vis = vis[0]
                    _maybe_save_image(os.path.join(cand_overlay_dir, fn), vis, save.overwrite)  # type: ignore
            else:
                record["step3"] = {"reason": "skipped"}

            rf.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            prev_pre, prev_name = pre, fn

        rf.close()
