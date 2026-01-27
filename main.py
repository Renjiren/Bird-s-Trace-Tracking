# main.py
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import birds_pipeline as bp
from preprocessing import PreprocessConfig
from camera_motion_compensation import CamMotionConfig
from candidate_generation import CandidateGenConfig


def parse_list(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser("birds pipeline (Step1~Step3, stage-wise saving)")

    # ---- data ----
    p.add_argument("--data_root", type=str, required=True, help="e.g. ./val")
    p.add_argument("--manifest", type=str, default="", help="COCO-like json; empty -> scan folders")
    p.add_argument("--videos", type=str, default="", help="comma-separated video names; empty=all")
    p.add_argument("--examples", action="store_true", help="use built-in EG_VIDEOS subset")
    p.add_argument("--every_n", type=int, default=1, help="sample every N frames (>=1)")

    # ---- run ----
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="all",
                   choices=["pre", "cam", "cand", "all"],
                   help="pre=Step1 only, cam=Step1+2, cand/all=Step1+2+3")

    # ---- saving ----
    p.add_argument("--no_save_images", action="store_true", help="disable saving stage images")
    p.add_argument("--save_every", type=int, default=1, help="save images every K processed frames (>=1)")
    p.add_argument("--overwrite", action="store_true")

    # ---- ablation ----
    p.add_argument("--ablate_no_cam_motion", action="store_true", help="skip Step2 warping (for debug/ablation)")

    # ---- minimal tweak knobs (optional) ----
    # Step1
    p.add_argument("--subtitle_mask_ratio", type=float, default=0.12)

    # Step3 (common)
    p.add_argument("--min_area", type=int, default=80)
    p.add_argument("--max_boxes", type=int, default=50)

    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    only_videos = bp.EG_VIDEOS if args.examples else parse_list(args.videos)
    manifest_path = args.manifest.strip() or None

    # --- configs (lean: rely on dataclass defaults) ---
    pre_cfg = PreprocessConfig(
        subtitle_mask_ratio=float(args.subtitle_mask_ratio),
    )

    cam_cfg = CamMotionConfig()  # defaults are robust; tweak in code only if needed

    cand_cfg = CandidateGenConfig(
        min_area=int(args.min_area),
        max_boxes=int(args.max_boxes),
    )

    save_cfg = bp.SaveConfig(
        save_images=(not args.no_save_images),
        save_every=max(1, int(args.save_every)),
        overwrite=bool(args.overwrite),
    )

    bp.run(
        data_root=args.data_root,
        out_root=args.out_dir,
        stage_arg=args.stage,
        pre_cfg=pre_cfg,
        cam_cfg=cam_cfg,
        cand_cfg=cand_cfg,
        manifest_path=manifest_path,
        only_videos=only_videos,
        every_n=max(1, int(args.every_n)),
        save=save_cfg,
        ablate_no_cam_motion=bool(args.ablate_no_cam_motion),
    )


if __name__ == "__main__":
    main()

