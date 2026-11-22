# -*- coding: utf-8 -*-
"""
demo_v7.py —— 在 v6 基础上，把判定层换成 judge_core.ParkingJudge

相对 v6 的变化（zones.json 格式不变，可直接沿用）：
  [判定] 脚点进出区域加滞回，边界抖动不再把计时清零
  [判定] 计时单位是"秒"，不再依赖 --vote 帧数；区域内漏检/遮挡 --gap 秒内不中断计时
  [判定] 位移超过车身对角线的 --move 倍 = 在骑行/推行，重新计时
  [判定] 冷却按"同区域 + 空间距离"，取代 80px 网格（网格在格子边界会失效）
  [判定] 先过滤类别再做跟踪，其他类别不会抢走 ebike/bicycle 的 ID
  [跟踪] 全局按 IoU 降序配对，结果与检测顺序无关；丢失的轨迹保留 --gap 秒
  [输出] 告警时保存带框截图到 <out>/evidence，events.csv 多一列"证据截图"
  [输出] 可选：企业微信推送（--notify，webhook 放环境变量 WECOM_WEBHOOK）
  [输入] 支持单张图片（解决 B1）：只标出每辆车落在哪个区域，不判定时长
  [输入] --stride N：每 N 帧处理 1 帧（树莓派上用，违停不需要逐帧）
  [可选] --person-model：加载 COCO 模型(类别0=person)，有人骑/扶着的车不计停放时间

用法：
    python demo_v7.py --source 视频.mp4 --model D:\\models\\ebike26.pt --zones zones.json --speed 10
    python demo_v7.py --source 图片.jpg --model D:\\models\\ebike26.pt --zones zones.json

时间参数说明：--speed 会把时间轴加速（演示用）。--gap / --rider-min 以"视频真实秒"为单位，
程序内部会自动乘以 speed；zones.json 里的 dwell / cooldown 与 v6 一样，是加速后的秒数。
"""
import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from judge_core import DEFAULT_VEHICLE_LABELS, JudgeConfig, ParkingJudge, WeComNotifier, foot, load_zones

CN = {"bicycle": "自行车", "motorcycle": "摩托", "ebike": "电动车", "tricycle": "三轮车"}
COLOR_OK = (0, 200, 0)         # 不在任何区域
COLOR_WAIT = (0, 215, 255)     # 在区域内但未达标
COLOR_RIDER = (200, 200, 200)  # 有人骑/扶着
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEVEL_NAME = {1: "一级（立即处理）", 2: "二级（尽快处理）", 3: "三级（例行处理）"}


# ---------------- 工具 ----------------
def imread(path):
    """cv2.imread 在 Windows 上读不了中文路径，用 imdecode 绕开"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def detect(model, frame, conf, iou, keep, names):
    """返回 boxes(list of xyxy), classes(list of int)；先按类别过滤，再交给跟踪"""
    res = model(frame, conf=conf, iou=iou, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return [], []
    xyxy = res.boxes.xyxy.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    boxes, classes = [], []
    for b, c in zip(xyxy, clss):
        if keep and names.get(int(c), str(int(c))) not in keep:
            continue
        boxes.append([float(v) for v in b])
        classes.append(int(c))
    return boxes, classes


def detect_persons(pmodel, frame, conf=0.4):
    res = pmodel(frame, conf=conf, classes=[0], verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    return [[float(v) for v in b] for b in res.boxes.xyxy.cpu().numpy()]


def near_zone(zones, box, margin):
    return max(z.signed_dist(foot(box)) for z in zones) >= -margin


def draw_zone(frame, z, alpha=0.12):
    poly = z.polygon.astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], z.color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [poly], True, z.color, 2)
    # 标签放在最靠上的顶点，避免和左上角计数重叠
    x, y = poly[poly[:, 1].argmin()]
    cv2.putText(frame, f"{z.name} L{z.level}", (int(x) + 8, int(y) + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, z.color, 2)


def draw_vehicle(frame, v, names):
    x1, y1, x2, y2 = map(int, v["box"])
    name = names.get(v["cls"], str(v["cls"]))
    z = v["zone"]
    if z is None:
        color, tag = COLOR_OK, ""
    elif v["rider"]:
        color, tag = COLOR_RIDER, "有人使用"
    elif v["warned"]:
        color, tag = z.color, f"L{z.level} 违停"
    else:
        color, tag = COLOR_WAIT, f"{z.name} {v['dwell']:.0f}s"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{CN.get(name, name)} #{v['id']} {tag}".strip(),
                (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    fx, fy = foot(v["box"])
    cv2.circle(frame, (int(fx), int(fy)), 4, color, -1)


# ---------------- 单张图片（B1） ----------------
def run_image(args, model, names, keep, out_dir):
    img = imread(args.source)
    if img is None:
        raise SystemExit(f"读不到图片：{args.source}")
    h, w = img.shape[:2]
    zones = load_zones(args.zones, w, h)
    boxes, classes = detect(model, img, args.conf, args.iou, keep, names)
    for z in zones:
        draw_zone(img, z)
    print(f"[图片] {w}x{h}，检测到 {len(boxes)} 辆；静态图无法判断停留时长，只标出落在哪个区域")
    for i, (b, c) in enumerate(zip(boxes, classes), 1):
        hits = [z for z in zones if z.signed_dist(foot(b)) >= 0]
        z = min(hits, key=lambda x: x.level) if hits else None
        name = names.get(c, str(c))
        draw_vehicle(img, {"box": b, "cls": c, "id": i, "zone": z, "rider": False,
                           "warned": False, "dwell": 0.0}, names)
        print(f"    #{i} {CN.get(name, name)} -> {z.name + ' (L%d)' % z.level if z else '区域外'}")
    out = out_dir / f"{Path(args.source).stem}_zones.jpg"
    imwrite(out, img)
    print(f"[图片] {out}")


# ---------------- 视频 / 摄像头 ----------------
def run_video(args, model, names, keep, out_dir):
    is_cam = str(args.source).isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"打不开：{args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    zones = load_zones(args.zones, w, h)
    print(f"[区域] 共 {len(zones)} 个：")
    for z in sorted(zones, key=lambda x: x.level):
        print(f"    L{z.level} {z.name}: 停留 ≥{z.dwell_s:g}s，冷却 {z.cooldown_s if z.cooldown_s is not None else '默认'}s")

    margin = max(2.0, args.margin_frac * w)
    cfg = JudgeConfig(enter_margin_px=margin, exit_margin_px=margin,
                      gap_s=args.gap * args.speed, move_frac=args.move,
                      cooldown_s=600.0, rider_min_s=args.rider_min * args.speed,
                      evidence_dir=str(out_dir / "evidence"))
    judge = ParkingJudge(zones, cfg)

    pmodel = YOLO(args.person_model) if args.person_model else None
    notifier = None
    if args.notify:
        notifier = WeComNotifier()
        if not notifier.webhook:
            print("[推送] 没有设置环境变量 WECOM_WEBHOOK，--notify 不会真正发送")

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(out_dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / args.stride), (w, h))
    print(f"[视频] {w}x{h} @ {fps:.1f}fps，时间加速 x{args.speed}，每 {args.stride} 帧处理 1 帧")

    rows, idx, wall0 = [], 0, time.time()
    while True:
        if not cap.grab():
            break
        if idx % args.stride:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = (time.time() - wall0) * args.speed if is_cam else idx / fps * args.speed
        idx += 1

        boxes, classes = detect(model, frame, args.conf, args.iou, keep, names)
        persons = None
        if pmodel is not None:  # 只在有车靠近禁停区时才跑人体检测，省算力
            persons = detect_persons(pmodel, frame) if any(
                near_zone(zones, b, margin) for b in boxes) else []

        events = judge.update(t, boxes, classes, persons=persons, frame=frame)

        for z in zones:
            draw_zone(frame, z)
        for v in judge.last_view:
            draw_vehicle(frame, v, names)
        for ev in events:
            name = names.get(ev["cls"], str(ev["cls"]))
            rows.append({"时间(s)": round(ev["time"], 1), "区域": ev["zone"], "等级": ev["level"],
                         "跟踪ID": ev["track_id"], "类别": name, "停留(s)": round(ev["dwell"], 1),
                         "处理方式": LEVEL_NAME.get(ev["level"], "未定义"),
                         "证据截图": ev["evidence"] or ""})
            print(f"[L{ev['level']} 报警] t={ev['time']:6.1f}s  {ev['zone']:6s} "
                  f"{CN.get(name, name)} ID={ev['track_id']} 停留{ev['dwell']:5.1f}s")
            if notifier:
                notifier.send(ev, now=time.time())
        cv2.putText(frame, f"events: {len(rows)}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    if notifier:
        notifier.flush_summary()

    cols = ["时间(s)", "区域", "等级", "跟踪ID", "类别", "停留(s)", "处理方式", "证据截图"]
    path = out_dir / "events.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # 没有事件也写表头，避免误看旧文件
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)
    if rows:
        print(f"\n[汇总] {len(rows)} 条事件 -> {path}")
        for lv in sorted({r["等级"] for r in rows}):
            print(f"    L{lv}: {sum(1 for r in rows if r['等级'] == lv)} 条")
    else:
        print("\n[汇总] 没有触发报警（可调大 --speed 加速时间，或检查区域配置）")
    if writer:
        print(f"[视频] {out_dir / 'demo.mp4'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="视频 / 图片路径，或 0 表示摄像头")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--keep", default=",".join(DEFAULT_VEHICLE_LABELS))
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--speed", type=float, default=1.0, help="时间加速倍数（演示用 10，真实用 1）")
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧处理 1 帧")
    ap.add_argument("--gap", type=float, default=3.0, help="区域内漏检/遮挡容忍时长（视频真实秒）")
    ap.add_argument("--move", type=float, default=0.35, help="位移超过车身对角线的该倍数视为移动")
    ap.add_argument("--margin-frac", type=float, default=0.004, help="进出区域的滞回宽度（占画面宽度比例）")
    ap.add_argument("--rider-min", type=float, default=1.0, help="有人持续多久才算在使用（视频真实秒）")
    ap.add_argument("--person-model", default=None, help="可选：COCO 模型路径，用于排除有人骑/扶着的车")
    ap.add_argument("--notify", action="store_true", help="企业微信推送（环境变量 WECOM_WEBHOOK）")
    ap.add_argument("--no-video", action="store_true", help="不输出带框视频")
    ap.add_argument("--out", default="runs/demo7")
    args = ap.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride 必须 >= 1")
    if not Path(args.zones).exists():
        raise SystemExit(f"找不到区域配置：{args.zones}")
    keep = {c.strip() for c in args.keep.split(",") if c.strip()} or None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    if Path(args.source).suffix.lower() in IMG_EXT:
        run_image(args, model, names, keep, out_dir)
    else:
        run_video(args, model, names, keep, out_dir)


if __name__ == "__main__":
    main()
