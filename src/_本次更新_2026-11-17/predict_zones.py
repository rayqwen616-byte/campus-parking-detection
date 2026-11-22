# -*- coding: utf-8 -*-
"""
predict_zones.py —— 一张图 → 自动生成 zones.json（禁停区多边形）· v2

改进：
    1. --min-area   可调的最小面积占比（默认 0.5%），过滤碎片
    2. --max-zones  最多输出几个区域（默认 8），避免杂乱
    3. --morph      形态学闭运算核大小（默认 7），补掉掩码空洞
    4. --all-contours  一个掩码里提取全部轮廓，而不是只取最大那个
    5. --level      覆盖 forbidden 的等级
    6. 输出带 _meta 统计，方便判断质量

用法：
    python predict_zones.py --model D:/models/zones.pt --source D:/zones_raw/B_001.jpg

输出：
    zones_auto.json      可直接给 demo_v8.py / webapp 使用
    zones_preview.jpg    可视化（红=禁停区，绿=通道）
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["forbidden", "passage"]
ZONE_CFG = {
    "forbidden": {"level": 1, "dwell": 10, "cooldown": 300, "color": [0, 0, 255]},
    "passage":   {"level": 3, "dwell": 180, "cooldown": 900, "color": [0, 200, 0]},
}


def imread_u(path):
    """兼容中文路径"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def clean_mask(mask, ksize=7):
    """掩码 → uint8，并用闭运算补空洞"""
    m = (mask > 0.5).astype(np.uint8) * 255
    if ksize and ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def mask_to_polygons(mask, w, h, eps_ratio=0.01, min_area_ratio=0.005,
                     morph=7, all_contours=False):
    """掩码 → 若干多边形（归一化坐标）"""
    m = clean_mask(mask, morph)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    if not all_contours:
        contours = [max(contours, key=cv2.contourArea)]

    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) / float(w * h) < min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        out.append([[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in approx])
    out.sort(key=lambda p: -cv2.contourArea(np.array(p, dtype=np.float32)))
    return out


def build_zones(result, w, h, args):
    """推理结果 → zones 列表"""
    names = result.names if isinstance(result.names, dict) else \
        {i: n for i, n in enumerate(result.names)}
    zones = []
    stat = {}
    if result.masks is None:
        return zones, stat

    masks = result.masks.data.cpu().numpy()
    classes = None
    if result.boxes is not None and result.boxes.cls is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)

    for i, mask in enumerate(masks):
        cls_id = int(classes[i]) if classes is not None and i < len(classes) else 0
        cls_name = str(names.get(cls_id,
                                 CLASSES[cls_id] if cls_id < len(CLASSES) else "forbidden"))
        cfg = ZONE_CFG.get(cls_name, ZONE_CFG["forbidden"]).copy()
        if cls_name == "forbidden" and args.level:
            cfg["level"] = args.level

        polys = mask_to_polygons(mask, w, h, args.eps, args.min_area,
                                 args.morph, args.all_contours)
        for k, poly in enumerate(polys):
            suffix = "" if len(polys) == 1 else "_%d" % (k + 1)
            zones.append({
                "name": ("禁停区" if cls_name == "forbidden" else "通道") + suffix,
                "level": cfg["level"],
                "dwell": cfg["dwell"],
                "cooldown": cfg["cooldown"],
                "color": cfg["color"],
                "points": poly,
                "_auto": {"class": cls_name, "vertices": len(poly)},
            })
        stat[cls_name] = stat.get(cls_name, 0) + len(polys)

    zones.sort(key=lambda z: z["level"])
    return zones[:args.max_zones], stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="zones_auto.json")
    ap.add_argument("--preview", default="zones_preview.jpg")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--eps", type=float, default=0.01, help="多边形简化程度，越大点越少")
    ap.add_argument("--min-area", type=float, default=0.005, help="最小面积占比")
    ap.add_argument("--max-zones", type=int, default=8)
    ap.add_argument("--morph", type=int, default=7, help="闭运算核，0 表示关闭")
    ap.add_argument("--all-contours", action="store_true", help="每个掩码提取全部轮廓")
    ap.add_argument("--level", type=int, default=0, help="覆盖 forbidden 等级（0=默认1）")
    args = ap.parse_args()

    src = Path(args.source)
    img = imread_u(src)
    if img is None:
        raise SystemExit("打不开图片：%s" % src)
    h, w = img.shape[:2]

    model = YOLO(args.model)
    res = model(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]

    zones, stat = build_zones(res, w, h, args)
    payload = {"zones": zones,
               "_meta": {"source": src.name, "size": [w, h],
                         "model": Path(args.model).name,
                         "count": len(zones), "by_class": stat}}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    vis = img.copy()
    for z in zones:
        pts = np.array([[int(x * w), int(y * h)] for x, y in z["points"]], dtype=np.int32)
        ov = vis.copy()
        cv2.fillPoly(ov, [pts], z["color"])
        cv2.addWeighted(ov, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [pts], True, z["color"], 3)
        cv2.putText(vis, "%s L%d" % (z["name"], z["level"]),
                    (max(0, pts[0][0] + 5), max(18, pts[0][1] + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, z["color"], 2)
    ok, buf = cv2.imencode(".jpg", vis)
    if ok:
        buf.tofile(args.preview)

    print("[结果] 识别出 %d 个区域  %s" % (len(zones), stat if stat else ""))
    for z in zones:
        print("    %-8s 等级L%d  %d 个顶点" % (z["name"], z["level"], len(z["points"])))
    print("[输出] %s" % args.out)
    print("[预览] %s" % args.preview)


if __name__ == "__main__":
    main()
