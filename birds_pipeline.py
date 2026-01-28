# birds_pipeline.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from preprocessing import (
    PreprocessConfig,
    preprocess_frame,
    ensure_bgr_u8,
    ensure_gray_u8,
    detect_subtitle_bbox,
    compute_log_feature,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")

EG_VIDEOS = [
    "Ac4002", "Ac4003", "An3004", "An3013",
    "An6013", "Ci2001", "Ci3001", "Pa1003", 
    "Gr5009", "Su2001", "Su2002", "Su2005"
]


@dataclass(frozen=True)
class SaveConfig:
    overwrite: bool = False
    save_overlay: bool = True


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_image(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _stem_int_or_name(fn: str) -> Tuple[int, str]:
    stem = os.path.splitext(os.path.basename(fn))[0]
    try:
        return int(stem), stem
    except Exception:
        return (1 << 30), stem


def list_videos(data_root: str) -> List[str]:
    return [
        name
        for name in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, name))
    ]


def list_frames(video_dir: str) -> List[str]:
    files = [fn for fn in os.listdir(video_dir) if _is_image(fn)]
    files.sort(key=_stem_int_or_name)
    return [os.path.join(video_dir, fn) for fn in files]


def _read_bgr(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return ensure_bgr_u8(bgr)


def _downscale_for_stats(bgr_u8: np.ndarray, max_width: int) -> np.ndarray:
    max_width = int(max(32, max_width))
    h, w = bgr_u8.shape[:2]
    if w <= max_width:
        return bgr_u8
    scale = max_width / float(w)
    nh = int(round(h * scale))
    return cv2.resize(bgr_u8, (max_width, nh), interpolation=cv2.INTER_AREA)


def _spec_percentiles(bgr_u8: np.ndarray, cfg: PreprocessConfig) -> Tuple[float, float]:
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    _, _, V = cv2.split(hsv)

    sigma = float(max(0.0, cfg.spec_blur_sigma))
    V_blur = cv2.GaussianBlur(V, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else V
    delta = cv2.subtract(V, V_blur)

    return float(np.percentile(V, 99.0)), float(np.percentile(delta, 99.5))


def _median_int(xs: Sequence[float], default: int) -> int:
    if not xs:
        return int(default)
    return int(round(float(np.median(np.asarray(xs, dtype=np.float32)))))


def _clip_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def _scale_bbox(bbox: Tuple[int, int, int, int], sx: float, sy: float) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def estimate_video_params(
    frame_paths: Sequence[str],
    base_cfg: PreprocessConfig,
    *,
    sample_n: int = 10,
    stats_max_width: int = 640,
    subtitle_persistent_ratio: float = 0.70,
    subtitle_min_votes: int = 3,
) -> Tuple[PreprocessConfig, Dict[str, Any]]:
    sample_n = max(1, int(sample_n))
    take = list(frame_paths[:sample_n])

    v_p99_list: List[float] = []
    d_p995_list: List[float] = []
    subtitle_boxes: List[Tuple[int, int, int, int]] = []
    edge_density_list: List[float] = []

    n_ok = 0
    for p in take:
        bgr = _read_bgr(p)
        if bgr is None:
            continue
        n_ok += 1

        bgr_s = _downscale_for_stats(bgr, stats_max_width)
        gray_s = ensure_gray_u8(bgr_s)

        v_p99, d_p995 = _spec_percentiles(bgr_s, base_cfg)
        v_p99_list.append(v_p99)
        d_p995_list.append(d_p995)

        bbox_s, _ = detect_subtitle_bbox(gray_s, base_cfg)
        if bbox_s is not None:
            sh, sw = gray_s.shape[:2]
            oh, ow = bgr.shape[:2]
            sx = ow / float(sw)
            sy = oh / float(sh)
            subtitle_boxes.append(_scale_bbox(bbox_s, sx, sy))

        log_u8 = compute_log_feature(gray_s, base_cfg)
        edge_density_list.append(float(np.mean(log_u8 >= 35)))

    v_min = _clip_int(_median_int(v_p99_list, base_cfg.spec_v_min), 180, 245)
    d_th = _clip_int(_median_int(d_p995_list, base_cfg.spec_delta_th), 6, 80)
    cfg = replace(base_cfg, spec_v_min=v_min, spec_delta_th=d_th)

    auto: Dict[str, Any] = {
        "sample_n": int(sample_n),
        "stats_max_width": int(stats_max_width),
        "samples_ok": int(n_ok),
        "spec": {
            "v_p99_median": float(np.median(v_p99_list)) if v_p99_list else None,
            "delta_p995_median": float(np.median(d_p995_list)) if d_p995_list else None,
            "v_min": int(v_min),
            "delta_th": int(d_th),
        },
        "edge_density_median": float(np.median(edge_density_list)) if edge_density_list else None,
    }

    if subtitle_boxes and n_ok > 0:
        ratio = len(subtitle_boxes) / float(n_ok)
        auto["subtitle_detect_ratio"] = float(ratio)

        if len(subtitle_boxes) >= int(subtitle_min_votes) and ratio >= float(subtitle_persistent_ratio):
            xs0 = [b[0] for b in subtitle_boxes]
            ys0 = [b[1] for b in subtitle_boxes]
            xs1 = [b[2] for b in subtitle_boxes]
            ys1 = [b[3] for b in subtitle_boxes]
            roi = (int(np.median(xs0)), int(np.median(ys0)), int(np.median(xs1)), int(np.median(ys1)))
            auto["subtitle_roi"] = roi
            cfg = replace(cfg, subtitle_roi=roi)
        else:
            auto["subtitle_roi"] = None
    else:
        auto["subtitle_detect_ratio"] = 0.0
        auto["subtitle_roi"] = None

    return cfg, auto


def _overlay_mask(bgr_u8: np.ndarray, mask_u8: np.ndarray, *, alpha: float = 0.55) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr_u8).copy()
    m0 = (mask_u8 == 0)
    if not np.any(m0):
        return bgr

    tint = np.zeros_like(bgr)
    tint[..., 2] = 255
    out = bgr.astype(np.float32)
    out[m0] = out[m0] * (1.0 - alpha) + tint[m0].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _maybe_write(path: str, img: np.ndarray, overwrite: bool) -> None:
    if (not overwrite) and os.path.exists(path):
        return
    cv2.imwrite(path, img)


def run_step1(
    *,
    data_root: str,
    out_root: str,
    pre_cfg: PreprocessConfig,
    only_videos: Optional[List[str]] = None,
    save: SaveConfig = SaveConfig(),
    sample_n: int = 10,
    stats_max_width: int = 640,
) -> None:
    _ensure_dir(out_root)

    videos_all = list_videos(data_root)
    if only_videos:
        wanted = set(only_videos)
        videos = [v for v in videos_all if v in wanted]
    else:
        videos = videos_all

    meta = {
        "source": {"type": "folder_traversal", "data_root": os.path.abspath(data_root)},
        "out_root": os.path.abspath(out_root),
        "save": asdict(save),
        "base_pre_cfg": asdict(pre_cfg),
        "sample_n": int(sample_n),
        "stats_max_width": int(stats_max_width),
        "only_videos": only_videos,
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

    global_summary: List[Dict[str, Any]] = []

    for v in videos:
        vdir = os.path.join(data_root, v)
        frame_paths = list_frames(vdir)
        if not frame_paths:
            continue

        first_path, last_path = frame_paths[0], frame_paths[-1]
        cfg_v, auto = estimate_video_params(
            frame_paths,
            pre_cfg,
            sample_n=sample_n,
            stats_max_width=stats_max_width,
        )

        vout = os.path.join(out_root, v)
        _ensure_dir(vout)

        video_params = {
            "video": v,
            "num_frames": int(len(frame_paths)),
            "source_dir": os.path.abspath(vdir),
            "frames_sorted": [os.path.basename(p) for p in frame_paths],
            "auto": auto,
            "pre_cfg_final": asdict(cfg_v),
        }
        with open(os.path.join(vout, "video_params.json"), "w", encoding="utf-8") as f:
            json.dump(video_params, f, ensure_ascii=False, indent=2, default=_json_default)

        def _process_one(tag: str, path: str) -> Dict[str, Any]:
            bgr = _read_bgr(path)
            if bgr is None:
                return {"tag": tag, "file": os.path.basename(path), "abs_path": os.path.abspath(path), "ok": False}

            pre = preprocess_frame(bgr, cfg_v)

            _maybe_write(os.path.join(vout, f"{tag}_smooth_bgr.jpg"), pre.smooth_bgr, save.overwrite)
            _maybe_write(os.path.join(vout, f"{tag}_valid_mask.png"), pre.valid_mask, save.overwrite)
            _maybe_write(os.path.join(vout, f"{tag}_spec_mask.png"), pre.spec_mask, save.overwrite)

            if save.save_overlay:
                _maybe_write(os.path.join(vout, f"{tag}_valid_overlay.jpg"), _overlay_mask(bgr, pre.valid_mask), save.overwrite)
                _maybe_write(os.path.join(vout, f"{tag}_spec_overlay.jpg"), _overlay_mask(bgr, pre.spec_mask), save.overwrite)

            h, w = pre.intensity.shape[:2]
            stats = {
                "H": int(h),
                "W": int(w),
                "valid_ratio": float(np.mean(pre.valid_mask > 0)),
                "spec_ratio": float(np.mean(pre.spec_mask == 0)),
                "log_edge_density": float(np.mean(pre.log >= 35)),
            }
            return {
                "tag": tag,
                "file": os.path.basename(path),
                "abs_path": os.path.abspath(path),
                "stats": stats,
                "debug": pre.debug,
                "ok": True,
            }

        first_info = _process_one("first", first_path)
        last_info = _process_one("last", last_path)

        summary_video = {
            "video": v,
            "num_frames": int(len(frame_paths)),
            "first": first_info,
            "last": last_info,
            "auto": auto,
        }
        with open(os.path.join(vout, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_video, f, ensure_ascii=False, indent=2, default=_json_default)

        global_summary.append(summary_video)

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2, default=_json_default)


