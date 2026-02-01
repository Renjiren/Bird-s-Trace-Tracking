# eval_cv.py
import os
import motmetrics as mm
import numpy as np

GT_DIR = "val_mot"
PRED_DIR = "output/track"

def load_mot_txt(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            items = line.strip().split(",")
            frame = int(items[0])
            obj_id = int(items[1])
            x, y, w, h = map(float, items[2:6])
            if frame not in data:
                data[frame] = []
            data[frame].append((obj_id, x, y, w, h))
    return data

def main():
    accs = []
    names = []
    for seq in os.listdir(GT_DIR):
        if not seq.endswith(".txt"):
            continue

        pred_path = os.path.join(PRED_DIR, seq)
        if not os.path.exists(pred_path):
            print(f"[SKIP] {seq} not predicted")
            continue   # ⭐ 关键：直接跳过

        print(f"Evaluating {seq}")
        gt = load_mot_txt(os.path.join(GT_DIR, seq))
        #pr = load_mot_txt(os.path.join(PRED_DIR, seq))
        pr = load_mot_txt(pred_path)

        acc = mm.MOTAccumulator(auto_id=True)

        frames = sorted(set(gt.keys()) | set(pr.keys()))
        for t in frames:
            gt_objs = gt.get(t, [])
            pr_objs = pr.get(t, [])

            gt_ids = [o[0] for o in gt_objs]
            pr_ids = [o[0] for o in pr_objs]

            gt_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in gt_objs])
            pr_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in pr_objs])

            if len(gt_boxes) == 0 or len(pr_boxes) == 0:
                acc.update(gt_ids, pr_ids, np.empty((len(gt_ids), len(pr_ids))))
                continue

            dists = mm.distances.iou_matrix(
                gt_boxes, pr_boxes, max_iou=0.5
            )
            acc.update(gt_ids, pr_ids, dists)

        accs.append(acc)
        names.append(seq.replace(".txt", ""))  # ⭐ 关键

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"],
        names=names
    )

    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

if __name__ == "__main__":
    main()