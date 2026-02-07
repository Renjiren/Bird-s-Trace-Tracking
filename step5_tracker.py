# step5_tracker.py
import numpy as np
import cv2


def iou(a, b):
    '''
    IoU: Intersection over Union
    '''
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


def ioa(inner, outer):
    '''
    IoA: Intersection over Area of the inner box (not union)
    Useful for checking if a small box is mostly contained within a larger box
    '''
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer

    ix2, iy2 = ix + iw, iy + ih
    ox2, oy2 = ox + ow, oy + oh

    x1 = max(ix, ox)
    y1 = max(iy, oy)
    x2 = min(ix2, ox2)
    y2 = min(iy2, oy2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    return inter / (iw * ih + 1e-6)


def suppress_tracks(tracks, iou_thr=0.8, ioa_thr=0.9, area_ratio_thr=0.6):
    """
    Remove duplicate tracks:
    1) near-identical tracks (IoU high)
    2) contained tracks (IoA high)
    """
    if not tracks:
        return tracks

    keep = [True] * len(tracks)
    boxes = [t.box for t in tracks]
    areas = [b[2] * b[3] for b in boxes]

    for i in range(len(tracks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(tracks)):
            if not keep[j]:
                continue

            bi, bj = boxes[i], boxes[j]
            ai, aj = areas[i], areas[j]

            iou_ij = iou(bi, bj)

            # Case 1: almost identical → keep stronger one
            if iou_ij > iou_thr:
                if tracks[i].hits >= tracks[j].hits:
                    keep[j] = False
                else:
                    keep[i] = False
                break

            # Case 2: containment → drop smaller
            if ai < aj:
                overlap = ioa(bi, bj) 
                if overlap > ioa_thr:
                    keep[i] = False
                    break

    return [t for k, t in zip(keep, tracks) if k]


# ---------------- Track ----------------
class Track:
    _next_id = 0

    def __init__(self, box, gray):
        self.id = Track._next_id
        Track._next_id += 1

        self.box = list(box)  # x,y,w,h (float ok)
        self.hits = 1
        self.miss = 0

        # LK points (using bbox center)
        x, y, w, h = box
        self.pts = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)
        self.prev_gray = gray.copy()

    def lk_step(self, curr_gray):
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        if next_pts is None or status is None:
            return False
        if status.size < 1 or status[0, 0] == 0:
            return False

        dx = float(next_pts[0, 0, 0] - self.pts[0, 0, 0])
        dy = float(next_pts[0, 0, 1] - self.pts[0, 0, 1])

        self.box[0] += dx
        self.box[1] += dy

        self.pts = next_pts
        self.prev_gray = curr_gray.copy()
        return True

    def update_with_det(self, det_box, gray):
        self.box = list(det_box)
        x, y, w, h = det_box
        self.pts = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)
        self.prev_gray = gray.copy()
        self.hits += 1
        self.miss = 0
        

# ---------------- Tracker ----------------
class Tracker:
    def __init__(self, iou_thr=0.3, max_age=5, min_hits=2):
        Track._next_id = 0

        self.tracks = []
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits

    def step(self, detections, prev_gray, curr_gray):
        used_det = set()

        # 1) LK track propagation
        for t in self.tracks:
            ok = t.lk_step(curr_gray)
            if ok:
                t.miss = 0
            else:
                t.miss += 1

        # 2) Detection correction (IoU matching)
        for t in self.tracks:
            best_iou = 0.0
            best_di = -1
            for di, d in enumerate(detections):
                if di in used_det:
                    continue

                v_iou = iou(t.box, d)
                if v_iou > best_iou:
                    best_iou = v_iou
                    best_di = di

            if best_iou >= self.iou_thr and best_di >= 0:
                t.update_with_det(detections[best_di], curr_gray)
                used_det.add(best_di)

        # 3) New detection → new track (with IoA suppression)
        for di, d in enumerate(detections):
            if di in used_det:
                continue

            suppressed = False
            for t in self.tracks:
             # if detection already tracked, don't create a new one
                if ioa(d, t.box) > 0.8:
                    suppressed = True
                    break

            if not suppressed:
                self.tracks.append(Track(d, curr_gray))

        # 4) Prune
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]

        # 5) Remove tracks contained in larger tracks
        self.tracks = suppress_tracks(self.tracks)

        # 6) Output
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                x, y, w, h = map(int, t.box)
                out.append((t.id, (x, y, w, h)))
        return out
