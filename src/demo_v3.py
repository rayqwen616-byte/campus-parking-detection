# -*- coding: utf-8 -*-
"""
demo_v3.py —— 针对「违停检测」做的三处推理优化

【优化1】NMS 的 IoU 阈值可调（解决一辆车出两个框）
【优化2】TTA 测试时增强（augment=True，多尺度 + 翻转）
【优化3】切片推理 slice（大图切块检测，小目标不再被缩没）

这三个都不需要重新训练，只改推理代码就能提升检出率。

用法：
    python demo_v3.py --source D:/xxx --model D:/models/ebike26.pt --keep ebike,bicycle --slice
"""
import argparse
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

CN = {"bicycle": "自行车", "motorcycle": "摩托", "ebike": "电动车",
      "tricycle": "三轮车", "vacant": "空位", "occupied": "占用",
      "unknown": "未知", "illegal_parking": "违停"}
COLORS = {"ebike": (0, 200, 0), "bicycle": (255, 140, 0), "tricycle": (0, 165, 255)}
DEFAULT_COLOR = (200, 200, 200)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_roi(text, w, h):
    if text == "full":
        return (0, 0, w, h)
    x1, y1, x2, y2 = (float(v) for v in text.split(","))
    return (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))


# ============ 【优化3】切片推理 ============
def detect_slice(model, frame, conf, iou, tiles=2, overlap=0.2):
    """
    把大图切成 tiles×tiles 块分别检测，再把结果拼回去。
    好处：每个小目标在块里被放大，更容易检出。
    坏处：推理次数变成 tiles² 倍，速度变慢。
    """
    H, W = frame.shape[:2]
    tw, th = W // tiles, H // tiles
    boxes, scores, clss = [], [], []

    for i in range(tiles):
        for j in range(tiles):
            x0 = max(int(j * tw - overlap * tw), 0)
            y0 = max(int(i * th - overlap * th), 0)
            x1 = min(int((j + 1) * tw + overlap * tw), W)
            y1 = min(int((i + 1) * th + overlap * th), H)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            res = model(crop, conf=conf, iou=iou, verbose=False)[0]
            if res.boxes is None:
                continue
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy().astype(int),
                               res.boxes.conf.cpu().numpy()):
                boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
                clss.append(int(c))
                scores.append(float(s))

    if not boxes:
        return [], [], []

    bt = torch.tensor(boxes, dtype=torch.float32)
    st = torch.tensor(scores, dtype=torch.float32)
    ct = torch.tensor(clss, dtype=torch.int64)

    keep_all = []
    for k in ct.unique():                       # 按类别分别做 NMS
        idx = (ct == k).nonzero(as_tuple=False).flatten()
        keep = nms(bt[idx], st[idx], iou)       # 【优化1】用可调的 iou
        keep_all.extend(idx[keep].tolist())

    keep_all.sort()
    return [boxes[i] for i in keep_all], [scores[i] for i in keep_all], [clss[i] for i in keep_all]


def draw(frame, boxes, scores, clss, names, keep, roi, show_count=True):
    n = 0
    counts = {}
    for b, s, c in zip(boxes, scores, clss):
        name = names.get(c, str(c))
        if keep and name not in keep:
            continue
        n += 1
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = map(int, b)
        color = COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{CN.get(name, name)} {s:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if roi:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 220, 255), 2)
    if show_count and counts:
        y = 28
        for name, cnt in counts.items():
            cv2.putText(frame, f"{CN.get(name, name)}: {cnt}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(name, DEFAULT_COLOR), 2)
            y += 26
    return n, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--keep", default="bicycle,motorcycle")
    ap.add_argument("--roi", default="full")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45, help="【优化1】NMS 的 IoU 阈值，越小去重越狠")
    ap.add_argument("--tta", action="store_true", help="【优化2】开启测试时增强")
    ap.add_argument("--slice", dest="use_slice", action="store_true", help="【优化3】开启切片推理")
    ap.add_argument("--out", default="runs/demo3")
    args = ap.parse_args()

    keep = {c.strip() for c in args.keep.split(",") if c.strip()} or None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    src = Path(args.source)
    is_cam = str(args.source).isdigit()
    print(f"[配置] conf={args.conf} iou={args.iou} tta={args.tta} slice={args.use_slice}")

    def run_one(frame):
        """统一入口：根据开关选择推理方式"""
        if args.use_slice:
            return detect_slice(model, frame, args.conf, args.iou)
        res = model(frame, conf=args.conf, iou=args.iou,
                    augment=args.tta, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return [], [], []
        return (res.boxes.xyxy.cpu().numpy().tolist(),
                res.boxes.conf.cpu().numpy().tolist(),
                res.boxes.cls.cpu().numpy().astype(int).tolist())

    # ---------------- 文件夹 ----------------
    if not is_cam and src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
        print(f"[batch] 共 {len(files)} 张")
        total = 0
        for i, p in enumerate(files, 1):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            roi = parse_roi(args.roi, w, h) if args.roi != "full" else None
            boxes, scores, clss = run_one(frame)
            n, _ = draw(frame, boxes, scores, clss, names, keep, roi)
            total += n
            cv2.imwrite(str(out_dir / ("result_" + p.name)), frame)
            if i % 10 == 0:
                print(f"  {i}/{len(files)}")
        print(f"[done] {len(files)} 张，共检出 {total} 个目标 -> {out_dir}")
        return

    # ---------------- 单图 ----------------
    if not is_cam and src.suffix.lower() in IMG_EXT:
        frame = cv2.imread(str(src))
        if frame is None:
            raise SystemExit(f"读不了图片：{src}")
        h, w = frame.shape[:2]
        roi = parse_roi(args.roi, w, h) if args.roi != "full" else None
        boxes, scores, clss = run_one(frame)
        n, counts = draw(frame, boxes, scores, clss, names, keep, roi)
        cv2.imwrite(str(out_dir / ("result_" + src.name)), frame)
        print(f"[done] 检出 {n} 个目标，明细 {counts}")
        return

    # ---------------- 摄像头 / 视频 ----------------
    cap = cv2.VideoCapture(int(args.source) if is_cam else str(src))
    if not cap.isOpened():
        raise SystemExit(f"打不开：{src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = parse_roi(args.roi, w, h)
    writer = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[video] {w}x{h} @ {fps:.1f}fps")
    idx, events = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores, clss = run_one(frame)
        n, _ = draw(frame, boxes, scores, clss, names, keep, roi)
        events += n
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    print(f"[done] {idx} 帧 -> {out_dir / 'demo.mp4'}")


if __name__ == "__main__":
    main()
