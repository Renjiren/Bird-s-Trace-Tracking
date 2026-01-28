# main.py
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import birds_pipeline as bp
from preprocessing import PreprocessConfig


def parse_list(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "birds - Step1 (per-video auto-tuned preprocessing probe: first & last frame outputs)"
    )

    p.add_argument("--data_root", type=str, required=True, help="e.g. ./val (each subfolder is a video)")
    p.add_argument("--videos", type=str, default="", help="comma-separated video names; empty=all")
    p.add_argument("--examples", action="store_true", help="use built-in EG_VIDEOS subset")

    p.add_argument("--out_dir", type=str, required=True, help="output folder for Step1 results (suggest: ./step1)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    p.add_argument("--no_overlay", action="store_true", help="do not save overlay visualization images")

    p.add_argument("--sample_n", type=int, default=10, help="how many first frames used for video-level auto params")
    p.add_argument("--stats_max_width", type=int, default=640, help="downscale width for stats (speed)")

    p.add_argument("--subtitle_mask_ratio", type=float, default=0.14)

    p.add_argument("--smooth_mode", type=str, default="bilateral", choices=["none", "bilateral"])
    p.add_argument("--bilateral_d", type=int, default=7)
    p.add_argument("--bilateral_sigma_color", type=float, default=35.0)
    p.add_argument("--bilateral_sigma_space", type=float, default=15.0)

    p.add_argument("--log_sigma", type=float, default=1.2)
    p.add_argument("--log_ksize", type=int, default=3)

    p.add_argument("--spec_blur_sigma", type=float, default=3.0)
    p.add_argument("--spec_v_min", type=int, default=220, help="base v_min (will be auto-tuned per video)")
    p.add_argument("--spec_delta_th", type=int, default=18, help="base delta threshold (will be auto-tuned per video)")

    p.add_argument("--spec_v_high", type=int, default=235)
    p.add_argument("--spec_s_low", type=int, default=55)

    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    only_videos = bp.EG_VIDEOS if args.examples else parse_list(args.videos)

    pre_cfg = PreprocessConfig(
        subtitle_mask_ratio=float(args.subtitle_mask_ratio),

        smooth_mode=args.smooth_mode,  # type: ignore
        bilateral_d=int(args.bilateral_d),
        bilateral_sigma_color=float(args.bilateral_sigma_color),
        bilateral_sigma_space=float(args.bilateral_sigma_space),

        log_blur_sigma=float(args.log_sigma),
        log_ksize=int(args.log_ksize),

        spec_blur_sigma=float(args.spec_blur_sigma),
        spec_v_min=int(args.spec_v_min),
        spec_delta_th=int(args.spec_delta_th),
        spec_v_high=int(args.spec_v_high),
        spec_s_low=int(args.spec_s_low),
    )

    save_cfg = bp.SaveConfig(
        overwrite=bool(args.overwrite),
        save_overlay=not bool(args.no_overlay),
    )

    bp.run_step1(
        data_root=args.data_root,
        out_root=args.out_dir,
        pre_cfg=pre_cfg,
        only_videos=only_videos,
        save=save_cfg,
        sample_n=int(args.sample_n),
        stats_max_width=int(args.stats_max_width),
    )


if __name__ == "__main__":
    main()

