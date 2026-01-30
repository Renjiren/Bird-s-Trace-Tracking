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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("birds pipeline (pre/motion/cand selectable)")

    p.add_argument("--data_root", type=str, required=True, help="val 根路径（每个视频一个子目录）")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--step", type=str, default="motion", choices=["pre", "motion", "cand"])

    # video selection
    p.add_argument("--video_set", type=str, default="eg", choices=["eg", "all"], help="选择 EG_VIDEOS 或跑全部")
    p.add_argument("--videos", type=str, default="", help="手动指定 video 列表（优先级最高），逗号分隔")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rng_seed", type=int, default=123)

    # -------- PRE params --------
    p.add_argument("--subtitle_roi_y0_ratio", type=float, default=0.74)
    p.add_argument("--spec_enable_mode", type=str, default="texture_only", choices=["always", "texture_only"])
    p.add_argument("--glare_fill_v_offset", type=int, default=12)
    p.add_argument("--glare_fill_dilate_ksize", type=int, default=11)
    p.add_argument("--pre_adapt_first_n_frames", type=int, default=10)
    p.add_argument("--pre_sample_k", type=int, default=5)
    p.add_argument("--pre_save_first_n", type=int, default=2)

    # -------- MOTION params --------
    p.add_argument("--motion_k_pairs", type=int, default=3, help="motion：每视频抽多少个相邻帧对")
    p.add_argument("--roi_mode", type=str, default="strips", choices=["corners", "strips", "corners+strips"])

    # ✅ 修复默认值覆盖 bug：默认继承 CamMotionConfig 默认值
    default_soft_spec = CamMotionConfig().use_soft_spec
    g1 = p.add_mutually_exclusive_group()
    g1.add_argument("--soft_spec", dest="use_soft_spec", action="store_true", help="启用 Step2 soft spec suppression")
    g1.add_argument("--no_soft_spec", dest="use_soft_spec", action="store_false", help="关闭 Step2 soft spec suppression")
    p.set_defaults(use_soft_spec=default_soft_spec)

    p.add_argument("--soft_spec_alpha", type=float, default=0.0)
    p.add_argument("--soft_spec_blur_sigma", type=float, default=3.0)
    p.add_argument("--soft_spec_max_ratio", type=float, default=0.25)

    # -------- CAND params --------
    p.add_argument("--cand_mad_k", type=float, default=8.0)
    p.add_argument("--cand_specular_weight", type=float, default=0.35)
    p.add_argument("--cand_min_area", type=int, default=80)
    p.add_argument("--cand_close_ksize", type=int, default=9)

    # ✅ 修复默认值覆盖 bug：默认继承 CandidateGenConfig 默认值
    default_bridge = CandidateGenConfig().use_bridge
    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--cand_bridge", dest="cand_use_bridge", action="store_true", help="启用 bridge")
    g2.add_argument("--cand_no_bridge", dest="cand_use_bridge", action="store_false", help="关闭 bridge")
    p.set_defaults(cand_use_bridge=default_bridge)

    p.add_argument("--cand_max_boxes", type=int, default=50)
    p.add_argument("--cand_save_full_step2_debug", action="store_true",
                   help="cand：在 debug_step3.json 里保存完整 step2.debug（会更大）")

    return p


def main() -> None:
    args = build_parser().parse_args()
    only_videos = parse_list(args.videos)

    # Step PRE config
    pre_cfg = PreprocessConfig(
        subtitle_roi_y0_ratio=float(args.subtitle_roi_y0_ratio),
        spec_enable_mode=args.spec_enable_mode,  # type: ignore
        glare_fill_v_offset=int(args.glare_fill_v_offset),
        glare_fill_dilate_ksize=int(args.glare_fill_dilate_ksize),
    )

    # Step MOTION config
    cam_cfg = CamMotionConfig(
        roi_mode=args.roi_mode,  # type: ignore
        use_soft_spec=bool(args.use_soft_spec),
        soft_spec_alpha=float(args.soft_spec_alpha),
        soft_spec_blur_sigma=float(args.soft_spec_blur_sigma),
        soft_spec_max_ratio=float(args.soft_spec_max_ratio),
    )

    if args.step == "pre":
        out_root = os.path.join(args.out_dir, "pre")
        os.makedirs(out_root, exist_ok=True)

        bp.run_step_pre(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
            sample_k=int(args.pre_sample_k),
            save_first_n=int(args.pre_save_first_n),
            adapt_first_n_frames=int(args.pre_adapt_first_n_frames),
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
            video_set=str(args.video_set),
            only_videos=only_videos,
            k_pairs_per_video=int(args.motion_k_pairs),
            overwrite=bool(args.overwrite),
            rng_seed=int(args.rng_seed),
        )
        return

    # cand
    cand_cfg = CandidateGenConfig(
        mad_k=float(args.cand_mad_k),
        specular_weight=float(args.cand_specular_weight),
        min_area=int(args.cand_min_area),
        close_ksize=int(args.cand_close_ksize),
        use_bridge=bool(args.cand_use_bridge),
        max_boxes=int(args.cand_max_boxes),
    )

    if args.step == "cand":
        out_root = os.path.join(args.out_dir, "cand")
        os.makedirs(out_root, exist_ok=True)

        bp.run_step_cand(
            data_root=args.data_root,
            out_root=out_root,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            cand_cfg=cand_cfg,
            video_set=str(args.video_set),
            only_videos=only_videos,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()


