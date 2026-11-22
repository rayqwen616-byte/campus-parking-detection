# -*- coding: utf-8 -*-
"""
demo_v5.py —— 在 v4 基础上加入【多帧投票 + 停留时长 + 告警冷却】

【本次改动 B】把"单帧判定"升级为"带时间维度的判定"

判定链条：
    检测框 → 跟踪 ID → 落脚点在禁停区内 → 连续 N 帧都满足 → 停留 ≥ 阈值 → 报警
                                                        ↓
                                            同一位置冷却期内不再重复报警

用法（演示，阈值调小）：
    python demo_v5.py --source 视频.mp4 --model 模型 --roi "..." --dwell 3 --vote 5 --cooldown 10

用法（真实场景）：
    python demo_v5.py --source 摄像头 --dwell 180 --vote 8 --cooldown 600
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CN = {"bicycle": "自行车", "motorcycle": "摩托", "ebike": "电动车", "tricycle": "三轮车"}
COLOR_OK = (0, 200, 0)        # 绿：不在禁停区
COLOR_WAIT = (0, 215, 255)    # 黄：在区域内，但还没满足条件
COLOR_WARN = (0, 0, 255)      # 红：确认违停
ROI_COLOR = (0, 0, 255)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============ 【新增】简易 IoU 跟踪器 ============
def iou(a, b):
    """两个框的 IoU"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class SimpleTracker:
    """
    最简跟踪：把这一帧的框，和上一帧的框按 IoU 配对。
    配上了 = 同一辆车；配不上 = 新车。
    工业上用 ByteTrack / BoT-SORT，原理相同，只是配对算法更复杂。
    """

    def __init__(self, iou_thres=0.3, max_miss=5):
        self.tracks = {}        # id -> [box, miss_count]
        self.next_id = 1
        self.iou_thres = iou_thres
        self.max_miss = max_miss

    def update(self, boxes):
        result = []
        used = set()
        for box in boxes:
            best_id, best_iou = None, 0
            for tid, (tbox, _) in self.tracks.items():
                if tid in used:
                    continue
                v = iou(box, tbox)
                if v > best_iou:
                    best_iou, best_id = v, tid
            if best_iou >= self.iou_thres:
                self.tracks[best_id] = [box, 0]
                used.add(best_id)
                result.append((box, best_id))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = [box, 0]
                used.add(tid)
                result.append((box, tid))

        # 没被匹配上的轨迹，miss 计数 +1，太久没出现就删掉
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_miss:
                    self.tracks.pop(tid)
        return result


def parse_roi(text, w, h):
    if text is None or text == "full":
        return None
    tokens = text.replace(";", " ").strip().split()
    if len(tokens) == 1 and len(tokens[0].split(",")) == 4:
        x1, y1, x2, y2 = (float(v) for v in tokens[0].split(","))
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        pts = [tuple(map(float, t.split(","))) for t in tokens]
    return np.array([(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32)


def point_in_poly(poly, pt):
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def draw_roi(frame, poly):
    if poly is None:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (0, 0, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [poly], True, ROI_COLOR, 2)
    cv2.putText(frame, "NO PARKING", (poly[0][0] + 8, poly[0][1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2)


def cell_key(cx, cy, size=80):
    """把画面切成网格，用于"同一位置"的冷却判断"""
    return f"{int(cx // size)}_{int(cy // size)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--keep", default="ebike,bicycle")
    ap.add_argument("--roi", default="full")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    # ↓↓↓ 本次新增的三个参数
    ap.add_argument("--dwell", type=float, default=3.0, help="停留多少秒算违停（演示 3，真实 180）")
    ap.add_argument("--vote", type=int, default=5, help="连续多少帧命中才算候选")
    ap.add_argument("--cooldown", type=float, default=10.0, help="同一位置多少秒内不重复报警")
    ap.add_argument("--out", default="runs/demo5")
    args = ap.parse_args()

    keep = {c.strip() for c in args.keep.split(",") if c.strip()} or None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    src = Path(args.source)
    is_cam = str(args.source).isdigit()

    # ---------------- 静态图模式：只画框，不做时间判定 ----------------
    def run_static(path, dst_name):
        frame = cv2.imread(str(path))
        if frame is None:
            return
        h, w = frame.shape[:2]
        poly = parse_roi(args.roi, w, h)
        res = model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
        for b, c, s in zip(res.boxes.xyxy.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int),
                           res.boxes.conf.cpu().numpy()):
            name = names.get(c, str(c))
            if keep and name not in keep:
                continue
            x1, y1, x2, y2 = map(int, b)
            inside = point_in_poly(poly, ((x1 + x2) / 2, y2))
            color = COLOR_WAIT if inside else COLOR_OK
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{CN.get(name, name)} {s:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        draw_roi(frame, poly)
        cv2.imwrite(str(out_dir / dst_name), frame)
        print(f"[static] -> {out_dir / dst_name}（静态图不做时间判定）")

    if not is_cam and src.suffix.lower() in IMG_EXT:
        run_static(src, "result_" + src.name)
        return
    if not is_cam and src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                run_static(p, "result_" + p.name)
        return

    # ---------------- 视频 / 摄像头：带时间判定 ----------------
    cap = cv2.VideoCapture(int(args.source) if is_cam else str(src))
    if not cap.isOpened():
        raise SystemExit(f"打不开：{src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poly = parse_roi(args.roi, w, h)
    writer = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"[video] {w}x{h} @ {fps:.1f}fps")
    print(f"[规则] 连续 {args.vote} 帧在区域内 + 停留 {args.dwell}s → 报警；"
          f"同位置 {args.cooldown}s 内不重复")

    # ===== 每个跟踪 ID 的状态 =====
    tracker = SimpleTracker()
    tracks = {}
    # ===== 每个网格位置的最近一次报警时间 =====
    last_alert = {}
    events = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        res = model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
        # 【改动】用自己的跟踪器，替代 model.track（省掉 lap 依赖）
        if res.boxes is not None and len(res.boxes) > 0:
            raw_boxes = res.boxes.xyxy.cpu().numpy()
            raw_cls = res.boxes.cls.cpu().numpy().astype(int)
            raw_conf = res.boxes.conf.cpu().numpy()
            tracked = tracker.update(raw_boxes)
            pairs = [(box, tid, int(c), float(s)) for (box, tid), c, s in
                     zip(tracked, raw_cls, raw_conf)]
        else:
            pairs = []

        active_ids = set()
        if pairs:
            for b, tid, c, s in pairs:
                name = names.get(c, str(c))
                if keep and name not in keep:
                    continue
                active_ids.add(tid)
                x1, y1, x2, y2 = map(int, b)
                foot = ((x1 + x2) / 2, y2)
                inside = point_in_poly(poly, foot)

                st = tracks.setdefault(tid, {"in_frames": 0, "t0": None,
                                             "warned": False, "cell": None})
                if inside:
                    st["in_frames"] += 1
                    if st["t0"] is None:
                        st["t0"] = t
                else:
                    # 离开区域：状态复位（但保留 warned，防止来回横跳重复报）
                    st["in_frames"] = 0
                    st["t0"] = None

                dwell = (t - st["t0"]) if st["t0"] is not None else 0
                cell = cell_key(foot[0], foot[1])
                st["cell"] = cell

                # ===== 三条件同时满足 =====
                voted = st["in_frames"] >= args.vote
                long_enough = dwell >= args.dwell
                cooled = (t - last_alert.get(cell, -9999)) >= args.cooldown

                if inside and voted and long_enough and cooled and not st["warned"]:
                    st["warned"] = True
                    last_alert[cell] = t
                    ev = {"时间(s)": round(t, 1), "跟踪ID": tid, "类别": name,
                          "停留(s)": round(dwell, 1), "位置格": cell,
                          "框": f"{x1},{y1},{x2},{y2}"}
                    events.append(ev)
                    print(f"[报警] t={t:6.1f}s  ID={tid:<4d} {CN.get(name, name):4s} "
                          f"停留{dwell:5.1f}s  格{cell}")

                # ===== 画框颜色：红=已报警，黄=在区域内待定，绿=正常 =====
                if st["warned"]:
                    color, tag = COLOR_WARN, "VIOLATION"
                elif inside:
                    color, tag = COLOR_WAIT, f"wait {dwell:.0f}s"
                else:
                    color, tag = COLOR_OK, ""
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{CN.get(name, name)} #{tid} {tag}",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, color, -1)

        # 清理长时间消失的 ID
        for tid in list(tracks):
            if tid not in active_ids:
                tracks[tid]["in_frames"] = 0
                tracks[tid]["t0"] = None

        draw_roi(frame, poly)
        cv2.putText(frame, f"events: {len(events)}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    # ===== 事件落盘 =====
    if events:
        csv_path = out_dir / "events.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            wtr = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            wtr.writeheader()
            wtr.writerows(events)
        print(f"\n[汇总] 共 {len(events)} 次报警 -> {csv_path}")
    else:
        print("\n[汇总] 没有触发报警（可调低 --dwell 或 --vote 再试）")
    print(f"[视频] {out_dir / 'demo.mp4'}")


if __name__ == "__main__":
    main()
