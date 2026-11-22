# -*- coding: utf-8 -*-
"""
demo_v6.py —— 在 v5 基础上加入【多区域 + 分级告警】

【本次改动 C】不再只有一个禁停区，而是多个区域，每个区域有自己的：
    - 名称（消防通道 / 楼道口 / 普通通道）
    - 等级（1 最高，3 最低）
    - 停留阈值（消防通道 10 秒就报，普通通道 180 秒）
    - 冷却时间
    - 颜色

配置文件：zones.json
用法：
    python demo_v6.py --source 视频.mp4 --model 模型 --zones zones.json
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CN = {"bicycle": "自行车", "motorcycle": "摩托", "ebike": "电动车", "tricycle": "三轮车"}
COLOR_OK = (0, 200, 0)        # 不在任何区域
COLOR_WAIT = (0, 215, 255)    # 在区域内但未达标
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEVEL_NAME = {1: "一级（立即处理）", 2: "二级（尽快处理）", 3: "三级（例行处理）"}


# ---------------- 基础工具 ----------------
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class SimpleTracker:
    """最简 IoU 跟踪器：这一帧的框与上一帧谁重叠最多，就是同一个目标"""

    def __init__(self, iou_thres=0.3, max_miss=5):
        self.tracks = {}
        self.next_id = 1
        self.iou_thres = iou_thres
        self.max_miss = max_miss

    def update(self, boxes):
        result, used = [], set()
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
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_miss:
                    self.tracks.pop(tid)
        return result


# ---------------- 区域 ----------------
class Zone:
    def __init__(self, cfg, w, h):
        self.name = cfg["name"]
        self.level = int(cfg.get("level", 3))
        self.dwell = float(cfg.get("dwell", 180))
        self.cooldown = float(cfg.get("cooldown", 600))
        self.color = tuple(cfg.get("color", [0, 0, 255]))
        pts = cfg["points"]
        self.poly = np.array([(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32)

    def contains(self, pt):
        return cv2.pointPolygonTest(self.poly, (float(pt[0]), float(pt[1])), False) >= 0

    def draw(self, frame, alpha=0.12):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.poly], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [self.poly], True, self.color, 2)
        x, y = self.poly[0]
        cv2.putText(frame, f"{self.name} L{self.level}", (x + 8, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.color, 2)


def load_zones(path, w, h):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [Zone(z, w, h) for z in cfg["zones"]]


def find_zone(zones, pt):
    """一个点可能落在多个区域，取等级最高（数字最小）的那个"""
    hit = None
    for z in zones:
        if z.contains(pt):
            if hit is None or z.level < hit.level:
                hit = z
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--keep", default="ebike,bicycle")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--vote", type=int, default=5)
    ap.add_argument("--speed", type=float, default=1.0, help="时间加速倍数（演示用 5 会把 60s 当 12s 跑）")
    ap.add_argument("--out", default="runs/demo6")
    args = ap.parse_args()

    keep = {c.strip() for c in args.keep.split(",") if c.strip()} or None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    src = Path(args.source)

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else str(src))
    if not cap.isOpened():
        raise SystemExit(f"打不开：{src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not Path(args.zones).exists():
        raise SystemExit(f"找不到区域配置：{args.zones}")
    zones = load_zones(args.zones, w, h)
    print(f"[区域] 共 {len(zones)} 个：")
    for z in sorted(zones, key=lambda x: x.level):
        print(f"    L{z.level} {z.name}: 停留 ≥{z.dwell}s，冷却 {z.cooldown}s")

    writer = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[视频] {w}x{h} @ {fps:.1f}fps，时间加速 x{args.speed}")

    tracker = SimpleTracker()
    states = {}          # (tid, zone_name) -> 状态
    last_alert = {}      # (zone_name, cell) -> 上次报警时间
    events = []
    idx = 0

    def cell_key(cx, cy, size=80):
        return f"{int(cx // size)}_{int(cy // size)}"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps * args.speed       # 演示时可以加速时间轴
        res = model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            pairs = [(b, tid, int(c), float(s))
                     for (b, tid), c, s in zip(tracker.update(boxes), clss, confs)]
        else:
            pairs = []

        active = set()
        for b, tid, c, conf in pairs:
            name = names.get(c, str(c))
            if keep and name not in keep:
                continue
            active.add(tid)
            x1, y1, x2, y2 = map(int, b)
            foot = ((x1 + x2) / 2, y2)
            zone = find_zone(zones, foot)

            if zone is None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OK, 2)
                cv2.putText(frame, f"{CN.get(name, name)} #{tid}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OK, 1)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, COLOR_OK, -1)
                continue

            key = (tid, zone.name)
            st = states.setdefault(key, {"in_frames": 0, "t0": None, "warned": False})
            st["in_frames"] += 1
            if st["t0"] is None:
                st["t0"] = t
            dwell = t - st["t0"]
            cell = cell_key(foot[0], foot[1])
            cooled = (t - last_alert.get((zone.name, cell), -99999)) >= zone.cooldown

            if (st["in_frames"] >= args.vote and dwell >= zone.dwell
                    and cooled and not st["warned"]):
                st["warned"] = True
                last_alert[(zone.name, cell)] = t
                ev = {"时间(s)": round(t, 1), "区域": zone.name, "等级": zone.level,
                      "跟踪ID": tid, "类别": name, "停留(s)": round(dwell, 1),
                      "处理方式": LEVEL_NAME.get(zone.level, "未定义")}
                events.append(ev)
                print(f"[L{zone.level} 报警] t={t:6.1f}s  {zone.name:6s} "
                      f"{CN.get(name, name)} ID={tid} 停留{dwell:5.1f}s")

            if st["warned"]:
                color, tag = zone.color, f"L{zone.level} 违停"
            else:
                color, tag = COLOR_WAIT, f"{zone.name} {dwell:.0f}s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{CN.get(name, name)} #{tid} {tag}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, color, -1)

        # 离开区域的轨迹复位
        for (tid, zname) in list(states):
            if tid not in active:
                st = states[(tid, zname)]
                st["in_frames"] = 0
                st["t0"] = None

        for z in zones:
            z.draw(frame)
        cv2.putText(frame, f"events: {len(events)}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    if events:
        path = out_dir / "events.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wtr = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            wtr.writeheader()
            wtr.writerows(events)
        print(f"\n[汇总] {len(events)} 条事件 -> {path}")
        by_level = {}
        for e in events:
            by_level[e["等级"]] = by_level.get(e["等级"], 0) + 1
        for lv in sorted(by_level):
            print(f"    L{lv}: {by_level[lv]} 条")
    else:
        print("\n[汇总] 没有触发报警（可调低 --speed 加速时间，或检查区域配置）")
    print(f"[视频] {out_dir / 'demo.mp4'}")


if __name__ == "__main__":
    main()
