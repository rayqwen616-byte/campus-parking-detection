# -*- coding: utf-8 -*-
"""demo_v7.py 端到端冒烟测试：用"假检测器"代替 ultralytics，视频/图片都是脚本现场合成的，
所以不需要权重、不需要联网。验证的是流程（读写、判定、CSV、证据截图、参数），不是检测精度。"""
import contextlib
import csv
import io
import os
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATE = {"n": 0, "mode": "bike"}


class _T:
    def __init__(self, a): self.a = np.array(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a
    def __len__(self): return len(self.a)


class _B:
    def __init__(self, boxes, cls):
        self.xyxy = _T(np.array(boxes, dtype=float).reshape(-1, 4))
        self.cls, self.conf = _T(cls), _T([0.9] * len(cls))
    def __len__(self): return len(self.cls)


class _R:
    def __init__(self, boxes, cls): self.boxes = _B(boxes, cls)


class FakeYOLO:
    def __init__(self, path):
        self.path, self.names = str(path), {0: "ebike", 1: "bicycle", 2: "car"}

    def __call__(self, frame, **kw):
        if "person" in self.path:  # 前 40 帧有人坐在车上
            return [_R([[110, 150, 300, 440]] if STATE["n"] <= 40 else [], [0])]
        STATE["n"] += 1
        if STATE["mode"] == "none":
            return [_R([], [])]
        # 一辆 ebike 静止在消防通道里 + 一辆 car（应被类别过滤掉）
        return [_R([[100, 200, 330, 470], [600, 100, 800, 300]], [0, 2])]


fake = types.ModuleType("ultralytics")
fake.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake
import demo_v7  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="smoke_v7_"))
VIDEO, IMAGE, ZONES = str(TMP / "synthetic.mp4"), TMP / "测试图.jpg", str(ROOT / "zones.json")

# 合成 960x540、10fps、90 帧的视频，以及一张中文文件名的图片
frame = (np.random.default_rng(0).integers(60, 160, (540, 960, 3))).astype(np.uint8)
vw = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 540))
for _ in range(90):
    vw.write(frame)
vw.release()
ok, buf = cv2.imencode(".jpg", frame)
buf.tofile(str(IMAGE))


def run(argv, mode="bike"):
    STATE.update(n=0, mode=mode)
    sys.argv = ["demo_v7.py"] + argv
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        demo_v7.main()
    return out.getvalue()


def rows(out):
    with open(Path(out) / "events.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


base = ["--source", VIDEO, "--model", "m.pt", "--zones", ZONES, "--speed", "10"]

o = str(TMP / "o1"); run(base + ["--out", o]); r = rows(o)
assert len(r) == 1 and r[0]["区域"] == "消防通道" and r[0]["类别"] == "ebike" and 9.9 <= float(r[0]["时间(s)"]) <= 10.5, r
assert os.path.getsize(r[0]["证据截图"]) > 0 and os.path.exists(o + "/demo.mp4")
print("1 基本流程 OK（car 被类别过滤，证据截图已保存）")

o = str(TMP / "o2"); run(base + ["--out", o, "--stride", "2"]); r = rows(o)
assert len(r) == 1 and 9.9 <= float(r[0]["时间(s)"]) <= 12.1, r
print("2 --stride 2 OK")

o = str(TMP / "o3"); run(base + ["--out", o, "--person-model", "person.pt"]); r = rows(o)
assert len(r) == 1 and 45 <= float(r[0]["时间(s)"]) <= 55, r
print("3 骑手过滤 OK（告警推迟到人离开后）")

o = str(TMP / "o4"); log = run(base + ["--out", o], mode="none")
assert rows(o) == [] and "没有触发报警" in log
print("4 无检测 OK（只写表头）")

o = str(TMP / "o5"); os.environ.pop("WECOM_WEBHOOK", None)
log = run(base + ["--out", o, "--notify", "--no-video"])
assert "WECOM_WEBHOOK" in log and len(rows(o)) == 1
print("5 --notify 无 webhook 不崩溃 OK")

o = TMP / "o6"
log = run(["--source", str(IMAGE), "--model", "m.pt", "--zones", ZONES, "--out", str(o)])
assert (o / "测试图_zones.jpg").exists() and "消防通道" in log, log
print("6 单张图片（中文路径）OK")
print("SMOKE PASSED")
