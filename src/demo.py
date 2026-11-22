# -*- coding: utf-8 -*-
"""
违停/车位检测演示：图片 / 文件夹 / 视频 / 摄像头 → 带框结果

用法：
    python demo.py --source D:/demo                    # 整个文件夹
    python demo.py --source D:/demo/a.jpg              # 单张图
    python demo.py --source D:/demo/a.mp4 --roi 0.1,0.5,0.9,1.0 --dwell 3
    python demo.py --source 0                          # 摄像头
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

CN = {"bicycle": "自行车", "motorcycle": "摩托", "ebike": "电动车",
      "tricycle": "三轮车", "vacant": "空位", "occupied": "占用",
      "unknown": "未知", "illegal_parking": "违停"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_roi(text, w, h):
    if text == "full":
        return (0, 0, w, h)
    x1, y1, x2, y2 = (float(v) for v in text.split(","))
    return (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))


def draw_boxes(frame, res, names, keep, roi):
    n = 0
    if res.boxes is None:
        return 0
    for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(),
                              res.boxes.cls.cpu().numpy().astype(int),
                              res.boxes.conf.cpu().numpy()):
        name = names.get(cls, str(cls))
        if keep and name not in keep:
            continue
        n += 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{CN.get(name, name)} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    if roi:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 220, 255), 2)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--keep", default="bicycle,motorcycle")
    ap.add_argument("--roi", default="full")
    ap.add_argument("--dwell", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()

    keep = {c.strip() for c in args.keep.split(",") if c.strip()} or None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    src = Path(args.source)
    is_cam = str(args.source).isdigit()

    # ---------------- 文件夹：批量出图 ----------------
    if not is_cam and src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
        print(f"[batch] 共 {len(files)} 张，开始推理...")
        total = 0
        for i, p in enumerate(files, 1):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            roi = parse_roi(args.roi, w, h) if args.roi != "full" else None
            res = model(frame, conf=args.conf, verbose=False)[0]
            total += draw_boxes(frame, res, names, keep, roi)
            cv2.imwrite(str(out_dir / ("result_" + p.name)), frame)
            if i % 10 == 0:
                print(f"  {i}/{len(files)}")
        print(f"[done] {len(files)} 张 -> {out_dir}，共检出 {total} 个目标")
        return

    # ---------------- 单张图 ----------------
    if not is_cam and src.suffix.lower() in IMG_EXT:
        frame = cv2.imread(str(src))
        if frame is None:
            raise SystemExit(f"读不了图片：{src}")
        h, w = frame.shape[:2]
        roi = parse_roi(args.roi, w, h) if args.roi != "full" else None
        res = model(frame, conf=args.conf, verbose=False)[0]
        n = draw_boxes(frame, res, names, keep, roi)
        dst = out_dir / ("result_" + src.name)
        cv2.imwrite(str(dst), frame)
        print(f"[done] {dst}，检出 {n} 个目标")
        return

    # ---------------- 视频 / 摄像头 ----------------
    cap = cv2.VideoCapture(int(args.source) if is_cam else str(src))
    if not cap.isOpened():
        raise SystemExit(f"打不开：{src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = parse_roi(args.roi, w, h)
    writer = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracks, events, idx = {}, [], 0
    print(f"[video] {w}x{h} @ {fps:.1f}fps，判定阈值 {args.dwell}s")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        res = model.track(frame, persist=True, conf=args.conf, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                     res.boxes.id.cpu().numpy().astype(int),
                                     res.boxes.cls.cpu().numpy().astype(int)):
                name = names.get(cls, str(cls))
                if keep and name not in keep:
                    continue
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, y2
                hit = roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]
                st = tracks.setdefault(tid, {"t0": None, "last": idx, "warned": False})
                st["last"] = idx
                if hit and st["t0"] is None:
                    st["t0"] = t
                if not hit:
                    st["t0"], st["warned"] = None, False
                dwell = (t - st["t0"]) if st["t0"] is not None else 0
                warn = dwell >= args.dwell
                if warn and not st["warned"]:
                    st["warned"] = True
                    events.append(tid)
                    print(f"[事件] t={t:.1f}s ID={tid} 停留 {dwell:.1f}s")
                color = (0, 0, 255) if warn else (0, 200, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{CN.get(name, name)} #{tid} {dwell:.0f}s",
                            (int(x1), max(int(y1) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for tid in list(tracks):
            if idx - tracks[tid]["last"] > int(fps * 2):
                tracks.pop(tid)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 220, 255), 2)
        cv2.putText(frame, f"events {len(events)}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"[done] {out_dir / 'demo.mp4'}，{len(events)} 个事件")


if __name__ == "__main__":
    main()
