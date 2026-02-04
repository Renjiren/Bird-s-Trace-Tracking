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

        # post-processing in ROI
        k = np.ones((3,3), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k, iterations=1)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=1)

        # Binarization
        fg = (roi > 0).astype(np.uint8)
        # Too few moving pixels in this box = noise, discard directly
        if fg.sum() < cfg.min_fg_pixels:
            continue

        # ---- 2. Distance Transform to find center ----
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            continue

        #cv2.imshow("roi_fg", fg * 255)
        #cv2.imshow("roi_dist", (dist / dist.max() * 255).astype(np.uint8))

        # Prospect Center (markers)
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
        
        # ---- 5. Each instance → bbox ----
        for lab in range(2, num_markers + 1):
            inst = (ws == lab).astype(np.uint8)
            # Dilate instance back to full fg (recover wings)
            inst = cv2.dilate(inst, np.ones((3,3), np.uint8), iterations=1)
            inst = cv2.bitwise_and(inst, fg)
            area = int(inst.sum())
            if area < cfg.min_area_ratio * H * W: 
                continue

            ys, xs = np.where(inst > 0)
            iy0, iy1 = ys.min(), ys.max()
            ix0, ix1 = xs.min(), xs.max()

            bx = x0 + ix0
            by = y0 + iy0
            bw = ix1 - ix0 + 1
            bh = iy1 - iy0 + 1

            # geometry filter (basic) 
            #if bw < 3 or bh < 3:
            #    continue
            if bw*bh > cfg.max_box_area_ratio * (H * W): 
                continue
            if min(bw, bh) < cfg.min_side:
                continue
            ar = bw / (bh + 1e-6) 
            if ar > cfg.aspect_ratio_max or ar < cfg.aspect_ratio_min:
                continue

            # ===== Birdness hard filters (no absolute size) =====

            fg_roi = mask_fg[by:by+bh, bx:bx+bw]
            fg_ratio = fg_roi.sum() / (255.0 * fg_roi.size + 1e-6)
            if fg_ratio < cfg.fg_ratio_min:
                continue

            fill = area / (bw * bh + 1e-6)   
            if fill < cfg.fill_ratio_min:
                continue

            if (bw*bh) > 0.25 * (H*W) and fill > 0.75:
                continue

            # Humans/cows tend to be more "vertical/thick", birds more "flat/expanded" — but this rule may mistakenly exclude certain poses
            #if bh > 3.5 * bw and (bw*bh) > 0.01 * (H*W):
            #    continue

            # spec constraint: if mostly in spec region, drop
            sm = spec_mask[by:by+bh, bx:bx+bw]
            if sm.size > 0:
                spec_frac = float(np.count_nonzero(sm == 0)) / float(sm.size)
                if spec_frac > cfg.spec_frac_max:
                    continue

            inst_boxes.append((area, bx, by, bw, bh))
        
        if not inst_boxes:
            # This step3 bbox is determined to be "not a bird"
            continue

        # Keep only the largest 1~2 instances (to prevent multiple boxes for one bird)
        inst_boxes.sort(key=lambda x: x[0], reverse=True)
        _, bx, by, bw, bh = inst_boxes[0]
        refined.append((bx, by, bw, bh))
     
    return refined