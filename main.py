# main.py
from __future__ import annotations
import argparse
from typing import Optional, List

from preprocessing import PreprocessConfig
from camera_motion import CamMotionConfig
from candidate_generation import candidateGenConfig
from birds_pipeline import (
    build_frames_by_video,
    run_preprocessing,
    run_camera_motion,
    run_candidate_generation_streaming,
)

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", required=True, help="数据集根目录：每个子目录=一个video，里面是帧图片")
    p.add_argument("--ann-json", default=None, help="可选：COCO-like json，提供 frame_id 排序")
    p.add_argument("--dataset", default="eg", choices=["eg", "all"], help="eg=仅跑指定小集合；all=跑全部")
    p.add_argument("--videos", nargs="*", default=None, help="进一步指定 video 子集（覆盖 dataset 选择）")

    p.add_argument("--sample-strategy", default="clips", choices=["clips", "every", "all"])
    p.add_argument("--num-clips", type=int, default=3)
    p.add_argument("--clip-len", type=int, default=10)
    p.add_argument("--every-n", type=int, default=10)

    p.add_argument("--overwrite", action="store_true", help="覆盖已有输出")


def add_step1_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--blur", default="none", choices=["none", "gaussian", "gaussian_median"])
    p.add_argument("--ksize", type=int, default=5, help="Gaussian kernel size (odd)")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--median-ksize", type=int, default=5)

    p.add_argument("--subtitle-mask", default="auto_bottom", choices=["none", "auto_bottom"])
    p.add_argument("--subtitle-ratio", type=float, default=0.12, help="底部屏蔽高度比例")


def add_step2_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="affine", choices=["affine", "homography"])
    p.add_argument("--nfeatures", type=int, default=2000)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--ratio-test", type=float, default=0.75)
    p.add_argument("--min-matches", type=int, default=30)
    p.add_argument("--ransac-thresh", type=float, default=3.0)
    p.add_argument("--max-iters", type=int, default=2000)

    # 门控参数（默认值是“最佳方案”的建议；你做消融时可改）
    p.add_argument("--gate-min-inliers", type=int, default=35)
    p.add_argument("--gate-min-trans", type=float, default=2.0)


def add_step3_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--thresh-mode", default="otsu", choices=["otsu", "mean_std", "percentile"])
    p.add_argument("--k-std", type=float, default=1.5)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--min-thresh", type=int, default=10)

    p.add_argument("--no-diff-median-suppress", action="store_true", help="关闭 diff 中位数抑制（消融）")

    p.add_argument("--open-ksize", type=int, default=3)
    p.add_argument("--open-iter", type=int, default=1)
    p.add_argument("--close-ksize", type=int, default=5)
    p.add_argument("--close-iter", type=int, default=1)

    p.add_argument("--min-area", type=int, default=25)
    p.add_argument("--max-area", type=int, default=30000)

    p.add_argument("--min-fill", type=float, default=0.02)
    p.add_argument("--max-fill", type=float, default=0.60)
    p.add_argument("--min-lap-var", type=float, default=12.0)
    p.add_argument("--bright-mean", type=int, default=200)
    p.add_argument("--bright-low-tex", type=float, default=18.0)

    p.add_argument("--max-keep", type=int, default=30)
    p.add_argument("--max-draw", type=int, default=30)
    p.add_argument("--min-score", type=float, default=0.0)

    p.add_argument("--no-save-diff", action="store_true")
    p.add_argument("--no-save-mask", action="store_true")
    p.add_argument("--no-save-overlay", action="store_true")


def build_pre_cfg(args) -> PreprocessConfig:
    k = int(args.ksize)
    if k % 2 == 0:
        k += 1
    return PreprocessConfig(
        blur_mode=args.blur,
        kernel_size=(k, k),
        sigma=float(args.sigma),
        median_ksize=int(args.median_ksize),
        subtitle_mask_mode=args.subtitle_mask,
        subtitle_mask_ratio=float(args.subtitle_ratio),
    )


def build_cam_cfg(args) -> CamMotionConfig:
    return CamMotionConfig(
        model=args.model,
        nfeatures=int(args.nfeatures),
        scale=float(args.scale),
        ratio_test=float(args.ratio_test),
        min_matches=int(args.min_matches),
        ransac_thresh=float(args.ransac_thresh),
        max_iters=int(args.max_iters),
        gate_min_inliers=int(args.gate_min_inliers),
        gate_min_translation_px=float(args.gate_min_trans),
    )


def build_candidate_gen_cfg(args) -> candidateGenConfig:
    return candidateGenConfig(
        thresh_mode=args.thresh_mode,
        k_std=float(args.k_std),
        percentile=float(args.percentile),
        min_thresh=int(args.min_thresh),
        use_diff_median_suppress=not args.no_diff_median_suppress,
        open_ksize=int(args.open_ksize),
        open_iter=int(args.open_iter),
        close_ksize=int(args.close_ksize),
        close_iter=int(args.close_iter),
        min_area=int(args.min_area),
        max_area=int(args.max_area),
        min_fill_ratio=float(args.min_fill),
        max_fill_ratio=float(args.max_fill),
        min_lap_var=float(args.min_lap_var),
        bright_mean=int(args.bright_mean),
        bright_low_tex=float(args.bright_low_tex),
        max_candidates_keep=int(args.max_keep),
        max_candidates_draw=int(args.max_draw),
        min_score_keep=float(args.min_score),
        save_diff=not args.no_save_diff,
        save_mask=not args.no_save_mask,
        save_overlay=not args.no_save_overlay,
    )


def main():
    parser = argparse.ArgumentParser("Birds Track - Stepwise Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Step1
    p1 = sub.add_parser("preprocessing", help="Step1: preprocess (gray/blur/subtitle mask) and save")
    add_common_args(p1)
    add_step1_args(p1)
    p1.add_argument("--pre-out", required=True, help="Step1 输出目录")

    # Step2
    p2 = sub.add_parser("camera_motion", help="Step2: align prev->curr using ORB+RANSAC and save")
    add_common_args(p2)
    add_step1_args(p2)  # 用于 motion gray 的 subtitle mask（同 Step1 保持一致）
    add_step2_args(p2)
    p2.add_argument("--pre-root", required=True, help="Step1 输出目录（输入）")
    p2.add_argument("--aligned-out", required=True, help="Step2 输出目录（对齐后的 prev + transforms json）")

    # Step3 (streaming 1-3)
    p3 = sub.add_parser("candidate_generation", help="Step3: run Step1-3 streaming and save Step3 outputs")
    add_common_args(p3)
    add_step1_args(p3)
    add_step2_args(p3)
    add_step3_args(p3)
    p3.add_argument("--out", required=True, help="Step3 输出根目录（diff/mask/overlay/candidates）")

    args = parser.parse_args()

    frames_by_video = build_frames_by_video(
        data_root=args.data_root,
        dataset=args.dataset,
        ann_json=args.ann_json,
        only_videos=args.videos,
        sample_strategy=args.sample_strategy,
        num_clips=args.num_clips,
        clip_len=args.clip_len,
        every_n=args.every_n,
    )

    if args.cmd == "preprocessing":
        pre_cfg = build_pre_cfg(args)
        run_preprocessing(args.data_root, args.pre_out, pre_cfg, frames_by_video, overwrite=args.overwrite)

    elif args.cmd == "camera_motion":
        pre_cfg = build_pre_cfg(args)
        cam_cfg = build_cam_cfg(args)
        run_camera_motion(
            data_root=args.data_root,
            pre_root=args.pre_root,
            aligned_root=args.aligned_out,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            frames_by_video=frames_by_video,
            overwrite=args.overwrite
        )

    elif args.cmd == "candidate_generation":
        pre_cfg = build_pre_cfg(args)
        cam_cfg = build_cam_cfg(args)
        candidate_gen_cfg = build_candidate_gen_cfg(args)
        run_candidate_generation_streaming(
            data_root=args.data_root,
            out_root=args.out,
            pre_cfg=pre_cfg,
            cam_cfg=cam_cfg,
            candidate_gen_cfg=candidate_gen_cfg,
            frames_by_video=frames_by_video,
            overwrite=args.overwrite
        )
    else:
        raise ValueError("Unknown cmd")


if __name__ == "__main__":
    main()
