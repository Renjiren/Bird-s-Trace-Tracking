# step4_refine.py
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class RefineConfig:
    min_area_ratio: float = 0.00005
    max_box_area_ratio: float = 0.3
    min_fg_pixels: int = 20
    fg_ratio_min: float = 0.2
    fill_ratio_min: float = 0.10
    spec_frac_max: float = 0.6
    min_side: int = 4
    aspect_ratio_min: float = 0.10
    aspect_ratio_max: float = 10.0
    CENTER_FRAC: float = 0.3  



def step4_refine(bgr, mask_fg, boxes_raw, spec_mask, cfg: RefineConfig):
    H, W = mask_fg.shape
    refined = []

    for (x, y, w, h) in boxes_raw:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x+w); y1 = min(H, y+h)
        roi = mask_fg[y0:y1, x0:x1].copy()
        if roi.size == 0:
            continue

        # post-processing in ROI - 1. ROI 内前景清理
        k = np.ones((3,3), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k, iterations=1)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=1)

        # 二值化
        fg = (roi > 0).astype(np.uint8)
        #这个框里动的像素太少 = 噪声，直接丢
        if fg.sum() < cfg.min_fg_pixels:
            continue

        # ---- 2. Distance Transform 找中心 ----
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            continue

        #cv2.imshow("roi_fg", fg * 255)
        #cv2.imshow("roi_dist", (dist / dist.max() * 255).astype(np.uint8))

        # 前景中心（markers）
        _, sure_fg = cv2.threshold(
            dist,
            cfg.CENTER_FRAC * dist.max(),  
            1,
            cv2.THRESH_BINARY
        )
        sure_fg = sure_fg.astype(np.uint8)

        # ---- 3. markers ----
        # ===== split gate: large & solid object =====
        
        bbox_area = (x1 - x0) * (y1 - y0)
        fg_area = fg.sum()

        fill_bbox = fg_area / (bbox_area + 1e-6)

        # 如果：
        # - bbox 足够大（相对全图）
        # - 且 fg 在 bbox 里是“实心的”
        # 就认为这是单一大物体（如大鸟），不做 split
        if bbox_area > 0.01 * (H * W) and fill_bbox > 0.25:
            refined.append((x0, y0, x1 - x0, y1 - y0))
            continue

        num_markers, markers = cv2.connectedComponents(sure_fg)
        if num_markers <= 1:
            markers = fg.copy()
            num_markers = 2
        else:
            markers = markers + 1
            markers[fg == 0] = 0

        # ---- 4. watershed ----
        roi_bgr = bgr[y0:y1, x0:x1]
        ws = cv2.watershed(roi_bgr, markers)

        inst_boxes = []
        
        # ---- 5. 每个实例 → bbox ----
        for lab in range(2, num_markers + 1):
            inst = (ws == lab).astype(np.uint8)
            # 把实例长回完整 fg（吃回翅膀）
            inst = cv2.dilate(inst, np.ones((3,3), np.uint8), iterations=1)
            inst = cv2.bitwise_and(inst, fg)
            area = int(inst.sum())
            if area < cfg.min_area_ratio * H * W: #面积太小 面积下限随分辨率变化 小噪声在大图里自动消失
                continue

            ys, xs = np.where(inst > 0)
            iy0, iy1 = ys.min(), ys.max()
            ix0, ix1 = xs.min(), xs.max()

            bx = x0 + ix0
            by = y0 + iy0
            bw = ix1 - ix0 + 1
            bh = iy1 - iy0 + 1

            # geometry filter (basic) - 6. 几何过滤
            #if bw < 3 or bh < 3:
            #    continue
            if bw*bh > cfg.max_box_area_ratio * (H * W):  # too huge 面积上限
                continue
            if min(bw, bh) < cfg.min_side: #删除太细
                continue
            ar = bw / (bh + 1e-6) #岩壁裂缝 / 电线 / 桥索 一刀切。
            if ar > cfg.aspect_ratio_max or ar < cfg.aspect_ratio_min:
                continue

            # ===== Birdness hard filters (no absolute size) =====

            # (A) bbox 内前景占比：鸟框里应有较多前景；城市/地面/人/牛常很“空”或边缘多
            fg_roi = mask_fg[by:by+bh, bx:bx+bw]
            fg_ratio = fg_roi.sum() / (255.0 * fg_roi.size + 1e-6)
            if fg_ratio < cfg.fg_ratio_min:
                continue

            # (B) 实例填充率：inst 在 bbox 里占多少（太稀碎/太空的不是鸟）
            fill = area / (bw * bh + 1e-6)   # area 是 inst.sum()
            if fill < cfg.fill_ratio_min:
                continue

            # (C) 极端大块且几乎全是前景：更像“整块背景在动/大物体”
            # 大鸟也可能大，但通常 fill 不会接近 1
            if (bw*bh) > 0.25 * (H*W) and fill > 0.75:
                continue

            # 人/牛常更“竖直/厚”，鸟更“扁/展开”——但这条可能误伤某些姿态
            #if bh > 3.5 * bw and (bw*bh) > 0.01 * (H*W):
            #    continue

            # spec constraint: if mostly in spec region, drop - 7. spec 约束
            sm = spec_mask[by:by+bh, bx:bx+bw]
            if sm.size > 0:
                spec_frac = float(np.count_nonzero(sm == 0)) / float(sm.size)
                if spec_frac > cfg.spec_frac_max:
                    continue

            inst_boxes.append((area, bx, by, bw, bh))
        
        if not inst_boxes:
            # 这个 step3 bbox 被判定为“不是鸟”
            continue

        # 只保留最大的 1~2 个实例（防一鸟多框）
        inst_boxes.sort(key=lambda x: x[0], reverse=True)
        _, bx, by, bw, bh = inst_boxes[0]
        refined.append((bx, by, bw, bh))
     
    return refined