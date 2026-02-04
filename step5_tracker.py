# step5_tracker.py
import numpy as np
import cv2


def iou(a, b):
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


# ---------------- Track ----------------
class Track:
    _next_id = 0

    def __init__(self, box, gray):
        self.id = Track._next_id
        Track._next_id += 1

        self.box = list(box)  # x,y,w,h
        self.hits = 1
        self.miss = 0
        self.time_since_update = 0

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

        if status[0, 0] == 0:
            return False

        dx = next_pts[0, 0, 0] - self.pts[0, 0, 0]
        dy = next_pts[0, 0, 1] - self.pts[0, 0, 1]

        # update bbox
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
        self.tracks = []
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits

    def step(self, detections, prev_gray, curr_gray):
        used_det = set()

        # 1) LK track propagation
        for t in self.tracks:
            ok = t.lk_step(curr_gray)
            if not ok:
                t.miss += 1

        # 2) Detection correction (IoU matching)
        for ti, t in enumerate(self.tracks):
            best_iou = 0
            best_di = -1
            for di, d in enumerate(detections):
                if di in used_det:
                    continue
                v = iou(t.box, d)
                if v > best_iou:
                    best_iou = v
                    best_di = di

            if best_iou >= self.iou_thr:
                t.update_with_det(detections[best_di], curr_gray)
                used_det.add(best_di)

        # 3) New detection → new track
        for di, d in enumerate(detections):
            if di not in used_det:
                self.tracks.append(Track(d, curr_gray))

        # 4) Prune 
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]

        # 5) Output
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                x, y, w, h = map(int, t.box)
                out.append((t.id, (x, y, w, h)))
        return out