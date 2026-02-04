# main.py
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import birds_pipeline as bp

from preprocessing import PreprocessConfig
from camera_motion_compensation import CamMotionConfig
from candidate_generation import CandidateGenConfig
from step4_refine import RefineConfig


def parse_list(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("birds pipeline (step1-3 minimal)")

    # dataset selection 
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--step", type=str, required=True, choices=["pre", "motion", "cand"])
    p.add_argument("--video_set", type=str, default="eg", choices=["eg", "all"])
    p.add_argument("--videos", type=str, default="", help="comma list, e.g. v1,v2")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rng_seed", type=int, default=123)

    # Step1 pre params
    p.add_argument("--subtitle_mask_mode", type=str, default="spec_roi", choices=["none", "spec_roi"])
    p.add_argument("--smooth_mode", type=str, default="bilateral", choices=["none", "bilateral"])
    p.add_argument("--spec_enable_mode", type=str, default="texture_only", choices=["always", "texture_only"])

    # Step2 motion params
    p.add_argument("--roi_mode", type=str, default="strips", choices=["strips", "corners", "corners+strips"])

    return p


def main() -> None:
    args = build_parser().parse_args()
    only_videos = parse_list(args.videos)

    pre_cfg = PreprocessConfig(
        subtitle_mask_mode=args.subtitle_mask_mode,  # type: ignore
        smooth_mode=args.smooth_mode,                # type: ignore
        spec_enable_mode=args.spec_enable_mode,      # type: ignore
    )

    cam_cfg = CamMotionConfig(roi_mode=args.roi_mode)  # type: ignore

    cand_cfg = CandidateGenConfig()

    if args.step == "pre":
        out_root = os.path.join(args.out_dir, "pre")
        os.makedirs(out_root, exist_ok=True)
        bp.run_step_pre(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            video_set=args.video_set,
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
            rng_seed=int(args.rng_seed),
        )
        return

    if args.step == "motion":
        out_root = os.path.join(args.out_dir, "motion")
        os.makedirs(out_root, exist_ok=True)
        bp.run_step_motion(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            video_set=args.video_set,
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
            rng_seed=int(args.rng_seed),
        )
        return

    if args.step == "cand":
        out_root = os.path.join(args.out_dir, "cand")
        os.makedirs(out_root, exist_ok=True)
        bp.run_step_cand(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            cand_cfg=cand_cfg,
            video_set=args.video_set,
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
        )
        return

    # refine
    if args.step == "refine":
        out_root = os.path.join(args.out_dir, "refine")
        os.makedirs(out_root, exist_ok=True)

        refine_cfg = RefineConfig()

        bp.run_step_refine(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            cand_cfg=cand_cfg,
            refine_cfg=refine_cfg,
            video_set=str(args.video_set),
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
        )
        return
    
    # track
    if args.step == "track":
        out_root = os.path.join(args.out_dir, "track")
        os.makedirs(out_root, exist_ok=True)

        refine_cfg = RefineConfig()
        bp.run_step_track(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            cand_cfg=cand_cfg,
            refine_cfg=refine_cfg,
            video_set=str(args.video_set),
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
        )
        return

if __name__ == "__main__":
    main()