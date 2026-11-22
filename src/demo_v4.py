# -*- coding: utf-8 -*-
"""
demo_v4.py —— 在 v3 基础上加入【多边形 ROI 判定】

【本次改动】把禁停区从"矩形"改成"任意多边形"，并区分区域内/区域外

用法（多边形）：
    python demo_v4.py --source 图或视频 --model 模型 ^
        --roi "0.05,0.55 0.42,0.42 0.95,0.70 0.90,0.98 0.08,0.98"

用法（矩形，兼容旧写法）：
    python demo_v4.py --source ... --roi 0.1,0.5,0.9,1.0

用法（全图，默认）：
    python demo_v4.py --source ...
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

CN = {"bicycle": "自行车", "motorcycle": "摩托", "ebike": "电动车",
      "tricycle": "三轮车", "vacant": "空位", "occupied": "占用",
      "unknown": "未知", "illegal_parking": "违停"}
COLORS = {"ebike": (0, 200, 0), "bicycle": (255, 140, 0), "tricycle": (0, 165, 255)}
OUT_COLOR = (160, 160, 160)      # 区域外的目标：灰色
ROI_COLOR = (0, 0, 255)          # 禁停区边框：红色
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============ 【改动】ROI 解析：支持矩形和多边形 ============
def parse_roi(text, w, h):
    """
    返回值：numpy 多边形顶点数组，或 None（表示全图）

    支持三种写法：
        full                     -> None
        0.1,0.5,0.9,1.0          -> 矩形（4 个数，兼容旧写法）
        "0.1,0.5 0.4,0.4 0.9,1"  -> 多边形（多个点，空格分隔）
    """
    if text is None or text == "full":
        return None

    text = text.replace(";", " ").strip()
    tokens = text.split()

    # 情况 A：一整串逗号分隔的 4 个数 → 矩形
    if len(tokens) == 1 and len(tokens[0].split(",")) == 4:
        x1, y1, x2, y2 = (float(v) for v in tokens[0].split(","))
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # 情况 B：多个 "x,y" → 多边形
    else:
        pts = []
        for t in tokens:
            x, y = t.split(",")
            pts.append((float(x), float(y)))

    if len(pts) < 3:
        raise SystemExit("ROI 至少要 3 个点才能围成区域")

    return np.array([(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32)


def point_in_poly(poly, pt):
    """判断点是否在多边形内部。poly 为 None 时视为全图（永远在里面）"""
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def draw_roi(frame, poly, alpha=0.18):
    """画禁停区：半透明填充 + 红色边框"""
    if poly is None:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (0, 0, 255))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)   # 半透明叠加
    cv2.polylines(frame, [poly], True, ROI_COLOR, 2)
    cv2.putText(frame, "NO PARKING", (poly[0][0] + 8, poly[0][1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2)


def detect_slice(model, frame, conf, iou, tiles=2, overlap=0.2):
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
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    bt = torch.tensor(boxes, dtype=torch.float32)
    st = torch.tensor(scores, dtype=torch.float32)
    ct = torch.tensor(clss, dtype=torch.int64)
    keep_all = []
    for k in ct.unique():
        idx = (ct == k).nonzero(as_tuple=False).flatten()
        keep_all.extend(idx[nms(bt[idx], st[idx], iou)].tolist())
    keep_all.sort()
    return (np.array([boxes[i] for i in keep_all]).reshape(-1, 4),
            np.array([scores[i] for i in keep_all]),
            np.array([clss[i] for i in keep_all], dtype=int))


def draw_all(frame, boxes, scores, clss, names, keep, poly):
    """
    画框 + 判定。
    【关键】用「落脚点」（框底边中点）判断是否在禁停区，
            而不是框中心——车停在哪里，看的是车轮压在哪。
    """
    n_all, n_in = 0, 0
    counts = {}
    for b, s, c in zip(boxes, scores, clss):
        name = names.get(c, str(c))
        if keep and name not in keep:
            continue
        n_all += 1
        x1, y1, x2, y2 = map(int, b)
        foot = ((x1 + x2) / 2, y2)          # 落脚点
        inside = point_in_poly(poly, foot)

        if inside:
            n_in += 1
            color = COLORS.get(name, (0, 200, 0))
            label = f"{CN.get(name, name)} {s:.2f} [违停区]"
            counts[name] = counts.get(name, 0) + 1
        else:
            color = OUT_COLOR
            label = f"{CN.get(name, name)} {s:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.circle(frame, (int(foot[0]), int(foot[1])), 5, color, -1)   # 落脚点

    # 统计面板
    cv2.putText(frame, f"total {n_all}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"in-zone {n_in}", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return n_all, n_in, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--keep", default="ebike,bicycle")
    ap.add_argument("--roi", default="full",
                    help='矩形 "x1,y1,x2,y2" 或多边形 "x1,y1 x2,y2 x3,y3"')
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--slice", dest="use_slice", action="store_true")
    ap.add_argument("--out", default="runs/demo4")
    args = ap.parse_args()

    keep = {c.strip() for c in args.keep.split(",") if c.strip()} or None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    src = Path(args.source)
    is_cam = str(args.source).isdigit()

    def run_one(frame):
        if args.use_slice:
            return detect_slice(model, frame, args.conf, args.iou)
        res = model(frame, conf=args.conf, iou=args.iou,
                    augment=args.tta, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return (np.zeros((0, 4)), np.zeros((0,)),
                    np.zeros((0,), dtype=int))
        return (res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int))

    # ---------------- 文件夹 ----------------
    if not is_cam and src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
        print(f"[batch] {len(files)} 张，ROI={args.roi}")
        tot_all = tot_in = 0
        for i, p in enumerate(files, 1):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            poly = parse_roi(args.roi, w, h)
            boxes, scores, clss = run_one(frame)
            draw_roi(frame, poly)
            n_all, n_in, _ = draw_all(frame, boxes, scores, clss, names, keep, poly)
            tot_all += n_all
            tot_in += n_in
            cv2.imwrite(str(out_dir / ("result_" + p.name)), frame)
            if i % 10 == 0:
                print(f"  {i}/{len(files)}")
        print(f"[done] {len(files)} 张：检出 {tot_all} 个，其中禁停区内 {tot_in} 个 -> {out_dir}")
        return

    # ---------------- 单图 ----------------
    if not is_cam and src.suffix.lower() in IMG_EXT:
        frame = cv2.imread(str(src))
        if frame is None:
            raise SystemExit(f"读不了图片：{src}")
        h, w = frame.shape[:2]
        poly = parse_roi(args.roi, w, h)
        boxes, scores, clss = run_one(frame)
        draw_roi(frame, poly)
        n_all, n_in, counts = draw_all(frame, boxes, scores, clss, names, keep, poly)
        cv2.imwrite(str(out_dir / ("result_" + src.name)), frame)
        print(f"[done] 检出 {n_all} 个，禁停区内 {n_in} 个，明细 {counts}")
        return

    # ---------------- 视频 / 摄像头 ----------------
    cap = cv2.VideoCapture(int(args.source) if is_cam else str(src))
    if not cap.isOpened():
        raise SystemExit(f"打不开：{src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poly = parse_roi(args.roi, w, h)
    writer = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[video] {w}x{h} @ {fps:.1f}fps，按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores, clss = run_one(frame)
        draw_roi(frame, poly)
        draw_all(frame, boxes, scores, clss, names, keep, poly)
        writer.write(frame)
        cv2.imshow("demo_v4 (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[done] -> {out_dir / 'demo.mp4'}")


if __name__ == "__main__":
    main()
