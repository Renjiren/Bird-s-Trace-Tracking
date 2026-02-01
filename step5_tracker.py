# step5_tracker.py
import numpy as np

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


def box_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def box_diag(b):
    x, y, w, h = b
    return float(np.hypot(w, h))


class Track:
    _next_id = 0

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1

        x, y, w, h = box
        self.cx = x + w / 2
        self.cy = y + h / 2
        self.vx = 0.0
        self.vy = 0.0
        self.w = w
        self.h = h

        self.age = 0
        self.hits = 1
        self.miss = 0

    def predict(self):
        #Kalman 预测
        self.cx += self.vx
        self.cy += self.vy
        self.age += 1

    def update(self, box):
        x, y, w, h = box
        nx = x + w / 2
        ny = y + h / 2

        #Kalman 更新
        self.vx = 0.7 * self.vx + 0.3 * (nx - self.cx)
        self.vy = 0.7 * self.vy + 0.3 * (ny - self.cy)

        self.cx, self.cy = nx, ny
        self.w, self.h = w, h

        self.hits += 1
        self.miss = 0

    def box(self):
        return (
            int(self.cx - self.w / 2),
            int(self.cy - self.h / 2),
            int(self.w),
            int(self.h),
        )


class Tracker:
    def __init__(self, iou_thr=0.3, max_age=5, min_hits=2, max_center_dist_frac=1.5, w_iou=0.7,  w_dist=0.3, birth_iou_block=0.10, birth_dist_frac=0.8, show_tentative=False):
        self.tracks = []
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits

        self.max_center_dist_frac = float(max_center_dist_frac)
        self.w_iou = float(w_iou)
        self.w_dist = float(w_dist)

        self.birth_iou_block = float(birth_iou_block)
        self.birth_dist_frac = float(birth_dist_frac)
        self.show_tentative = bool(show_tentative)

    def _cost(self, t_box, det_box):
        """
        cost 越小越好：
        - IoU 越大越好  -> (1 - iou)
        - 距离越小越好 -> dist_norm
        """
        v_iou = iou(t_box, det_box)

        tcx, tcy = box_center(t_box)
        dcx, dcy = box_center(det_box)
        dist = float(np.hypot(tcx - dcx, tcy - dcy))

        # 用 track 的对角线做归一化，避免大框/小框尺度不一致
        d = max(1e-6, box_diag(t_box))
        dist_norm = dist / d

        return self.w_iou * (1.0 - v_iou) + self.w_dist * dist_norm, v_iou, dist, d


    def step(self, detections):
        # predict
        for t in self.tracks:
            t.predict()

        if not self.tracks:
            # 没 track：全变成新 track
            for d in detections:
                self.tracks.append(Track(d))
            return []

        if not detections:
            # 没 detection：全部 miss
            for t in self.tracks:
                t.miss += 1
            self.tracks = [t for t in self.tracks if t.miss <= self.max_age]
            return [(t.id, t.box()) for t in self.tracks if t.hits >= self.min_hits]

        # 2) 构建所有可行配对（加 gating）
        pairs = []  # (cost, ti, di)
        for ti, t in enumerate(self.tracks):
            tb = t.box()
            for di, d in enumerate(detections):
                cost, v_iou, dist, diag = self._cost(tb, d)

                # gating 1：IoU 太小直接不考虑（你原本就是靠 iou_thr）
                if v_iou < self.iou_thr:
                    continue

                # gating 2：中心距离太远不考虑（解决“远处噪声框把 track 拉走”）
                if dist > self.max_center_dist_frac * diag:
                    continue

                pairs.append((cost, ti, di))

        # 3) 代价从小到大做“全局贪心”（比你原来的逐 track 贪心更稳）
        pairs.sort(key=lambda x: x[0])

        used_t = set()
        used_d = set()
        matches = []  # (ti, di)

        for cost, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))

        # 4) update matched
        for ti, di in matches:
            self.tracks[ti].update(detections[di])

        # 5) unmatched tracks -> miss
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.miss += 1

        # 6) unmatched dets -> new tracks(with birth gate)
        for di, d in enumerate(detections):
            if di in used_d:
                continue

            # --- birth gate：避免“同一只鸟身上不断冒新 track” ---
            ok_birth = True
            dcx, dcy = box_center(d)

            for t in self.tracks:
                tb = t.box()
                # 如果很像（IoU 有点重叠）就别生娃
                if iou(tb, d) > self.birth_iou_block:
                    ok_birth = False
                    break

                # 如果很近（中心很近）也别生娃
                tcx, tcy = box_center(tb)
                dist = float(np.hypot(tcx - dcx, tcy - dcy))
                diag = max(1e-6, box_diag(tb))
                if dist < self.birth_dist_frac * diag:
                    ok_birth = False
                    break

            if ok_birth:
                self.tracks.append(Track(d))

        # 7) prune dead tracks
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]

        # 8) output confirmed tracks only
        out = []
        for t in self.tracks:
            if self.show_tentative:
                out.append((t.id, t.box()))          # 调试：全部都画出来
            else:
                if t.hits >= self.min_hits:
                    out.append((t.id, t.box()))      # 正式：只画 confirmed
        return out