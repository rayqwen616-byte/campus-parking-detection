# -*- coding: utf-8 -*-
"""
webapp.py —— 单图识别网页演示（v9）

前端来自 Kimi 的方案（web/index.html），后端重写为直接调用 judge_core 的 Zone，
保证「网页单图」与「视频判定」使用同一套区域实现（单一真相源）。

启动：
    D:\\yolo_env\\Scripts\\python.exe webapp.py
    D:\\yolo_env\\Scripts\\python.exe webapp.py --port 8000 --model D:\\models\\ebike26.pt

浏览器打开 http://127.0.0.1:8000

接口：
    GET  /             网页
    GET  /api/health   健康检查
    POST /api/detect   {image: dataURL, confidence: 0.25, zones: [...]}
    POST /api/zones    {image: dataURL, confidence: 0.25}   需要 models/zones.pt
"""
import argparse
import binascii
import base64
import io
import json
import math
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps

from judge_core import DEFAULT_VEHICLE_LABELS, Zone, resolve_keep_labels
from predict_zones import mask_to_polygons

BASE = Path(__file__).resolve().parent
MAX_BYTES = 18 * 1024 * 1024
MAX_PIXELS = 40_000_000
CN = {"bicycle": "自行车", "cycle": "自行车", "bike": "自行车",
      "ebike": "电动车", "e-bike": "电动车", "electric_bike": "电动车",
      "motorcycle": "电动车/摩托车", "motorbike": "电动车/摩托车",
      "tricycle": "三轮车", "person": "人"}


def decode_data_url(data_url):
    """dataURL -> BGR ndarray；用 PIL 读，兼容中文与 EXIF 方向"""
    if not isinstance(data_url, str):
        raise ValueError("图片格式不正确")
    header, sep, encoded = data_url.partition(",")
    if not sep or header.lower() not in {
            "data:image/jpeg;base64", "data:image/jpg;base64",
            "data:image/png;base64", "data:image/webp;base64"}:
        raise ValueError("只支持 JPEG、PNG 或 WEBP 图片")
    if len(encoded) > ((MAX_BYTES + 2) // 3) * 4:
        raise ValueError("图片太大")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片数据不是有效的 Base64") from exc
    if len(raw) > MAX_BYTES:
        raise ValueError("图片太大")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("图片内容格式与支持的格式不符")
            if source.width <= 0 or source.height <= 0 or source.width * source.height > MAX_PIXELS:
                raise ValueError("图片像素过大")
            img = ImageOps.exif_transpose(source).convert("RGB")
            img.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法解码图片，请换用有效的 JPEG、PNG 或 WEBP 文件") from exc
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def encode_jpeg(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("结果图编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def zones_from_payload(items, w, h):
    """前端传来的区域列表 -> judge_core.Zone（与视频判定同一套类）"""
    if items is None:
        return []
    if not isinstance(items, list) or len(items) > 10:
        raise ValueError("区域必须是列表且不超过 10 个")
    zones = []
    seen_names = set()
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("区域必须是对象")
        name = str(it.get("name", "禁停区")).strip()[:40] or "禁停区"
        if name in seen_names:
            raise ValueError("区域名称不能重复")
        seen_names.add(name)
        level = int(it.get("level", 1))
        if level not in (1, 2, 3):
            raise ValueError("区域等级必须是 1、2 或 3")
        pts = it.get("points")
        if not isinstance(pts, list) or not 3 <= len(pts) <= 32:
            raise ValueError("每个区域需要 3 至 32 个顶点")
        arr = []
        for p in pts:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError("顶点格式必须是 [x, y]")
            x, y = float(p[0]), float(p[1])
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("顶点坐标必须是 0 到 1 的数字")
            arr.append((x * w, y * h))
        color = tuple(int(c) for c in it.get("color", (0, 0, 255)))
        if len(color) != 3 or any(not 0 <= c <= 255 for c in color):
            raise ValueError("颜色必须是 3 个 0 到 255 的数")
        dwell = float(it.get("dwell", 180))
        cooldown = float(it.get("cooldown", 600))
        if not math.isfinite(dwell) or dwell <= 0:
            raise ValueError("停留阈值必须是大于 0 的有限数字")
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError("冷却时间必须是大于或等于 0 的有限数字")
        zones.append(Zone(name, level, dwell,
                          np.array(arr, dtype=np.float32),
                          cooldown, color))
    return zones


def find_zone(zones, pt):
    """脚点落在多个区域时取等级最小（最严重）的"""
    hits = [z for z in zones if z.signed_dist(pt) >= 0]
    return min(hits, key=lambda z: z.level) if hits else None


def draw(frame, dets, zones):
    vis = frame.copy()
    for z in zones:
        poly = z.polygon.astype(np.int32)
        ov = vis.copy()
        cv2.fillPoly(ov, [poly], z.color)
        cv2.addWeighted(ov, 0.12, vis, 0.88, 0, vis)
        cv2.polylines(vis, [poly], True, z.color, 2)
        x, y = int(poly[0][0]), int(poly[0][1])
        cv2.putText(vis, f"{z.name} L{z.level}", (max(0, x + 5), max(18, y + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, z.color, 2)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = d["_color"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = "%s %.2f" % (CN.get(d["model_label"], d["model_label"]), d["confidence"])
        if d["zone"]:
            text += " [%s]" % d["zone"]
        cv2.putText(vis, text, (x1, max(y1 - 7, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(vis, (int(d["foot"][0]), int(d["foot"][1])), 4, color, -1)
    return vis


class State:
    model = None
    seg_model = None
    model_path = None
    keep = set(DEFAULT_VEHICLE_LABELS)
    inference_lock = threading.Lock()


def detect(image, confidence, zones, keep=None):
    h, w = image.shape[:2]
    keep = State.keep if keep is None else keep
    with State.inference_lock:
        res = State.model(image, conf=float(confidence), iou=0.45, verbose=False)[0]
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}
    dets = []
    if res.boxes is not None:
        for box, cid, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.cls.cpu().numpy().astype(int),
                                  res.boxes.conf.cpu().numpy()):
            label = str(names.get(int(cid), int(cid)))
            if keep and label not in keep:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            foot = ((x1 + x2) / 2.0, float(y2))
            zone = find_zone(zones, foot)
            dets.append({
                "label": CN.get(label, label),
                "model_label": label,
                "confidence": round(float(conf), 4),
                "box": [x1, y1, x2, y2],
                "foot": [round(foot[0], 1), round(foot[1], 1)],
                "status": "inside" if zone else "outside",
                "zone": zone.name if zone else None,
                "level": zone.level if zone else None,
                "_color": zone.color if zone else (60, 200, 60),
            })
    inside = sum(1 for d in dets if d["status"] == "inside")
    return dets, {"total": len(dets), "inside": inside, "outside": len(dets) - inside}


def zones_from_segmentation(res, w, h):
    zones_out = []
    if res.masks is None:
        return zones_out
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}
    classes = (res.boxes.cls.cpu().numpy().astype(int)
               if res.boxes is not None else [])
    for i, mask in enumerate(res.masks.data.cpu().numpy()):
        cid = int(classes[i]) if i < len(classes) else -1
        if str(names.get(cid, "")) != "forbidden":
            continue
        for poly in mask_to_polygons(mask, w, h, 0.01, 0.005, 7, True):
            zones_out.append({
                "name": f"禁停区{len(zones_out) + 1}",
                "level": 1,
                "points": poly,
            })
    return zones_out


class Handler(BaseHTTPRequestHandler):
    server_version = "CampusBikeV10/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            return self._json(200, {
                "ok": True,
                "model_loaded": State.model is not None,
                "model": Path(State.model_path).name if State.model_path else None,
                "keep": sorted(State.keep),
                "zone_model_available": State.seg_model is not None,
                "zone_model": Path(getattr(State.seg_model, "ckpt_path", "zones.pt")).name if State.seg_model else None,
            })
        if path == "/":
            path = "/index.html"
        web_root = (BASE / "web").resolve()
        f = (web_root / path.lstrip("/")).resolve()
        try:
            f.relative_to(web_root)
        except ValueError:
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        if not f.is_file():
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        ctype = {"html": "text/html; charset=utf-8",
                 "js": "text/javascript; charset=utf-8",
                 "css": "text/css; charset=utf-8"}.get(f.suffix.lstrip("."),
                                                       "application/octet-stream")
        return self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/api/detect", "/api/zones"):
            return self._json(404, {"ok": False, "error": "未知接口"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BYTES * 2:
                raise ValueError("请求体大小不合法")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("请求必须使用 application/json")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("请求体未完整接收")
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            image = decode_data_url(payload.get("image"))
            conf = float(payload.get("confidence", 0.25))
            if not math.isfinite(conf) or not 0 < conf < 1:
                raise ValueError("置信度必须在 0 到 1 之间")

            if route == "/api/zones":
                if State.seg_model is None:
                    raise ValueError("未加载区域分割模型（需要 models/zones.pt）")
                h, w = image.shape[:2]
                with State.inference_lock:
                    res = State.seg_model(image, conf=max(conf, 0.25), verbose=False)[0]
                zones_out = zones_from_segmentation(res, w, h)
                return self._json(200, {"ok": True, "zones": zones_out,
                                        "count": len(zones_out)})

            h, w = image.shape[:2]
            t0 = time.perf_counter()
            zones = zones_from_payload(payload.get("zones"), w, h)
            dets, summary = detect(image, conf, zones)
            vis = draw(image, dets, zones)
            for d in dets:
                d.pop("_color", None)
            return self._json(200, {
                "ok": True, "width": w, "height": h,
                "model": Path(State.model_path).name if State.model_path else None,
                "detections": dets, "summary": summary,
                "annotated_image": encode_jpeg(vis),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
        except (ValueError, KeyError, TypeError, json.JSONDecodeError,
                binascii.Error, OSError, OverflowError) as e:
            return self._json(400, {"ok": False, "error": str(e)})
        except Exception as e:
            print("[请求错误] %r" % (e,))
            return self._json(500, {"ok": False, "error": "服务器处理失败，请检查服务日志"})


def pick_zone_model(explicit):
    """自动找区域分割模型：models/zones.pt → D:/models/zones.pt → 最新训练结果"""
    if explicit:
        return Path(explicit) if Path(explicit).is_file() else None
    env_model = os.environ.get("CAMPUS_ZONES_MODEL")
    if env_model and Path(env_model).is_file():
        return Path(env_model)
    for c in (BASE / "models" / "zones.pt",
              Path("D:/models/zones.pt"),
              BASE.parent / "models" / "zones.pt"):
        if c.is_file():
            return c
    cands = sorted(BASE.glob("runs/segment/runs_zones/*/weights/best.pt"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def pick_model(explicit):
    if explicit:
        return Path(explicit)
    env_model = os.environ.get("CAMPUS_BIKE_MODEL")
    if env_model and Path(env_model).is_file():
        return Path(env_model)
    for c in (BASE / "models" / "ebike26.pt",
              BASE.parent / "models" / "ebike26.pt",
              Path("D:/models/ebike26.pt"),
              BASE / "yolo26n.pt"):
        if c.is_file():
            return c
    return Path("yolo11n.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--zone-model", default=None, help="可选：区域分割模型")
    ap.add_argument("--keep", default="auto",
                    help="auto=保留模型现有的常见非机动车类别；也可填写逗号分隔的类别名")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    from ultralytics import YOLO

    State.model_path = pick_model(args.model)
    print("[模型] %s" % State.model_path)
    State.model = YOLO(str(State.model_path))
    names = State.model.names
    try:
        State.keep = resolve_keep_labels(args.keep, names)
    except ValueError as exc:
        ap.error(str(exc))
    if not State.keep:
        ap.error("所选模型没有可用的白名单类别；请检查模型类别或显式设置 --keep")
    print("[类别] %s" % (list(names.values()) if isinstance(names, dict) else names))
    print("[保留类别] %s" % ", ".join(sorted(State.keep)))

    zm = pick_zone_model(args.zone_model)
    if zm:
        State.seg_model = YOLO(str(zm))
        print("[区域模型] %s  ← 「自动识别区域」按钮可用" % zm)
    else:
        print("[区域模型] 未找到 zones.pt，「自动识别区域」按钮会提示未加载")

    print("[服务] http://%s:%d  (Ctrl+C 退出)" % (args.host, args.port))
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
