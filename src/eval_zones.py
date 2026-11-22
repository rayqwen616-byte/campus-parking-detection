# -*- coding: utf-8 -*-
"""
eval_zones.py —— 区域分割模型的定量评测（不看训练 mAP，看真实多边形吻合度）

思路：
    对每张有 labelme 标注的图跑一次预测，把预测掩码转成多边形，
    再与人工标注的多边形按 IoU 匹配，统计：
        · 平均 IoU（best match）
        · 命中率 IoU >= 阈值（默认 0.5）
        · 精确率（预测出的区域里有多少是对的）
        · 漏检（GT 有、预测没有）

用法：
    python eval_zones.py --model D:/models/zones.pt --src D:/zones_raw --conf 0.25

输出：
    终端表格 + eval_zones.csv
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from predict_zones import imread_u, mask_to_polygons

CLASSES = ["forbidden", "passage"]


def poly_mask(poly, w, h):
    """归一化多边形 -> 二值掩码"""
    m = np.zeros((h, w), np.uint8)
    pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def iou_mask(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def load_gt(json_path, w, h):
    """读 labelme json，返回 {class: [归一化多边形, ...]}"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = {}
    for sh in data.get("shapes", []):
        label = str(sh.get("label", "")).strip().lower()
        if label not in CLASSES:
            continue
        pts = sh.get("points") or []
        if len(pts) < 3:
            continue
        out.setdefault(label, []).append([[float(p[0]) / w, float(p[1]) / h] for p in pts])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, help="labelme 标注目录（含 .json）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--hit-iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.005)
    ap.add_argument("--out", default="eval_zones.csv")
    args = ap.parse_args()

    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    src = Path(args.src)

    rows, ious_all, tp, fp, fn = [], [], 0, 0, 0
    for jp in sorted(src.glob("*.json")):
        img = imread_u(src / jp.name.replace(".json", ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(jp, w, h)

        res = model(img, conf=args.conf, verbose=False)[0]
        pred = {}
        if res.masks is not None:
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
            for i, m in enumerate(res.masks.data.cpu().numpy()):
                label = str(names.get(int(cls[i]), cls[i])) if len(cls) > i else "forbidden"
                poly = mask_to_polygons(m, w, h, min_area_ratio=args.min_area,
                                        all_contours=True)
                for p in poly:
                    pred.setdefault(label, []).append(p)

        for cname in CLASSES:
            gt_list = gt.get(cname, [])
            pr_list = pred.get(cname, [])
            if not gt_list and not pr_list:
                continue

            gt_m = [poly_mask(p, w, h) for p in gt_list] if gt_list else []
            pr_m = [poly_mask(p, w, h) for p in pr_list] if pr_list else []

            # 每个 GT 找最匹配的预测
            best_for_gt = []
            for g in gt_m:
                bi = max((iou_mask(g, p) for p in pr_m), default=0.0)
                best_for_gt.append(bi)

            # 每个预测找最匹配的 GT（算精确率）
            best_for_pred = []
            for p in pr_m:
                bi = max((iou_mask(g, p) for g in gt_m), default=0.0)
                best_for_pred.append(bi)

            hit = sum(1 for v in best_for_gt if v >= args.hit_iou)
            tp_c = sum(1 for v in best_for_pred if v >= args.hit_iou)
            fp_c = len(pr_m) - tp_c
            fn_c = len(gt_m) - hit

            tp += tp_c; fp += fp_c; fn += fn_c
            ious_all.extend(best_for_gt)

            rows.append({
                "image": jp.name.replace(".json", ".jpg"),
                "class": cname,
                "gt": len(gt_m),
                "pred": len(pr_m),
                "hit": hit,
                "mean_iou": round(float(np.mean(best_for_gt)), 4) if best_for_gt else 0.0,
                "max_iou": round(float(np.max(best_for_gt)), 4) if best_for_gt else 0.0,
            })

    if not rows:
        print("没有可评测的样本")
        return

    print(f"{'图片':<14}{'类别':<12}{'GT':>4}{'预测':>5}{'命中':>5}{'平均IoU':>10}{'最高IoU':>10}")
    print("-" * 62)
    for r in rows:
        print(f"{r['image']:<14}{r['class']:<12}{r['gt']:>4}{r['pred']:>5}"
              f"{r['hit']:>5}{r['mean_iou']:>10.3f}{r['max_iou']:>10.3f}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("-" * 62)
    print(f"平均 IoU      : {np.mean(ious_all):.4f}")
    print(f"IoU>=阈值 命中 : {tp} / {tp + fn}")
    print(f"精确率        : {precision:.3f}")
    print(f"召回率        : {recall:.3f}")
    print(f"F1            : {f1:.3f}")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n明细已存：{args.out}")


if __name__ == "__main__":
    main()
