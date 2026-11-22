"""judge_core.py - 违停判定层改进版（独立模块，不依赖 ultralytics）

对应技术简报里的 B1~B7 之外的"设计层"问题，逐项修复：
  1. 脚点在多边形边界抖动 -> 滞回（进入/退出用不同的有符号距离阈值）
  2. 连续 N 帧投票依赖帧率、一次漏检就清零 -> 全部改为"秒"，并允许 gap_s 内的漏检/遮挡
  3. 网格冷却在格子边界会失效 -> 改为"同区域 + 空间距离"冷却
  4. 贪心 IoU 跟踪与检测顺序有关 -> 全局按 IoU 降序配对，丢失轨迹保留 max_lost_s 秒
  5. 骑行/推行/临时停靠被当成违停 -> 静止检测（位移超过阈值则重置计时）+ 骑手检测（可选）
  6. 只有 events.csv，没有证据 -> 告警瞬间保存带框截图（用 imencode，避开 Windows 中文路径的 cv2.imwrite 失败）
  7. B7 无推送 -> 企业微信群机器人（限频、失败不抛异常、3 级合并成日报）
  8. "Recall>=0.90"没有证据 -> 事件级评测函数（precision / recall / 平均延迟 / 误报每小时）

用法（接到你现有的 demo_v6.py 里，替换 SimpleTracker + 判定部分）:
    zones = load_zones("zones.json", W, H)
    judge = ParkingJudge(zones, JudgeConfig(evidence_dir="evidence"))
    ...每处理一帧:
    events = judge.update(t_seconds, boxes_xyxy, class_ids, persons=None, frame=frame)

运行自测: python judge_core.py
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_VEHICLE_LABELS = ("ebike", "bicycle")


# ----------------------------------------------------------------- geometry
def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def foot(b):
    return ((b[0] + b[2]) / 2.0, float(b[3]))


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def diag(b) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


# -------------------------------------------------------------------- zones
@dataclass
class Zone:
    name: str
    level: int          # 1 = 最严重
    dwell_s: float      # 该等级的停留阈值（秒）
    polygon: np.ndarray  # (N,2) float32, 像素坐标
    cooldown_s: float | None = None  # 该区域的重复告警冷却；None 则用 JudgeConfig.cooldown_s
    color: tuple = (0, 0, 255)       # BGR，仅用于绘制

    def signed_dist(self, pt) -> float:
        """>0 在多边形内（到边界的像素距离），<0 在外。"""
        return cv2.pointPolygonTest(self.polygon, (float(pt[0]), float(pt[1])), True)


def load_zones(path: str, w: int, h: int):
    """读 zones.json（与 demo_v6 / predict_zones.py 的格式一致）：
        {"zones": [{"name","level","dwell","cooldown","color":[B,G,R],"points":[[x,y],...]}]}
    也兼容 polygon / dwell_s / cooldown_s 的写法。坐标全部 <=1 时按归一化坐标换算成像素。
    配置有错直接报错，不静默跳过（对应 B3）。
    """
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    items = raw["zones"] if isinstance(raw, dict) else raw
    zones = []
    for z in items:
        try:
            pts = np.array(z["points"] if "points" in z else z["polygon"], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                raise ValueError("多边形至少 3 个 (x,y) 点")
            if pts.max() <= 1.0:
                pts = pts * np.array([w, h], dtype=np.float32)
            dwell = z["dwell"] if "dwell" in z else z["dwell_s"]
            cool = z.get("cooldown", z.get("cooldown_s"))
            zones.append(Zone(str(z["name"]), int(z.get("level", 3)), float(dwell), pts,
                              None if cool is None else float(cool),
                              tuple(int(c) for c in z.get("color", (0, 0, 255)))))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"zones.json 中有无效区域 {z!r}: {e!r}") from e
    if not zones:
        raise ValueError("zones.json 没有任何区域")
    return zones


# ------------------------------------------------------------------ tracker
class Tracker:
    """全局贪心 IoU 跟踪：所有 (检测,轨迹) 配对按 IoU 降序接受，结果与检测顺序无关。
    类别不参与匹配（ebike/bicycle 置信度翻转很常见），保存最新类别。
    丢失的轨迹保留 max_lost_s 秒，遮挡恢复后仍是同一个 id。
    """

    def __init__(self, iou_thres: float = 0.3, max_lost_s: float = 3.0):
        self.iou_thres, self.max_lost_s = iou_thres, max_lost_s
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, t: float, boxes, classes):
        pairs = []
        for di, b in enumerate(boxes):
            for tid, tr in self.tracks.items():
                v = iou(b, tr["box"])
                if v >= self.iou_thres:
                    pairs.append((v, di, tid))
        pairs.sort(reverse=True)
        ids = [None] * len(boxes)
        used_t = set()
        for v, di, tid in pairs:
            if ids[di] is not None or tid in used_t:
                continue
            ids[di] = tid
            used_t.add(tid)
            self.tracks[tid].update(box=boxes[di], cls=classes[di], last_t=t)
        for di, b in enumerate(boxes):
            if ids[di] is None:
                ids[di] = self.next_id
                self.tracks[self.next_id] = {"box": b, "cls": classes[di], "last_t": t}
                self.next_id += 1
        for tid in [k for k, v in self.tracks.items() if t - v["last_t"] > self.max_lost_s]:
            del self.tracks[tid]
        return ids


# -------------------------------------------------------------------- judge
@dataclass
class JudgeConfig:
    enter_margin_px: float = 4.0     # 脚点需深入多边形这么多像素才算"进入"
    exit_margin_px: float = 4.0      # 脚点需越出这么多像素才算"离开"
    gap_s: float = 3.0               # 区域内漏检/遮挡容忍时长，期间计时不中断
    move_frac: float = 0.35          # 中心位移超过 box 对角线的这个比例 -> 视为在移动，重置计时
    cooldown_s: float = 120.0        # 同区域同位置的重复告警抑制时长
    cooldown_radius_frac: float = 0.6  # "同位置" = 中心距离 < 该比例 * box 对角线
    rider_overlap: float = 0.30      # 人框与车框的交集 / 车框面积 >= 此值 -> 判为有人骑/扶
    rider_min_s: float = 1.0         # 有人持续这么久才算"在使用"，路人一闪而过不重置
    evidence_dir: str | None = None  # 设置后，告警时保存带框截图


class _ZState:
    __slots__ = ("inside", "enter_t", "last_inside_t", "anchor_c", "warned")

    def __init__(self):
        self.inside = False
        self.enter_t = None
        self.last_inside_t = -1e18
        self.anchor_c = None
        self.warned = False


def _save_jpg(path: str, img) -> bool:
    """cv2.imwrite 在 Windows 上遇到非 ASCII 路径会静默失败，用 imencode + tofile。"""
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)
    return bool(ok)


class ParkingJudge:
    def __init__(self, zones, cfg: JudgeConfig | None = None, tracker: Tracker | None = None):
        self.zones = list(zones)
        self.cfg = cfg or JudgeConfig()
        self.tracker = tracker or Tracker(max_lost_s=self.cfg.gap_s)
        self.state: dict[int, dict[str, _ZState]] = {}
        self.rider_since: dict[int, float] = {}
        self.alerts: deque = deque()  # (zone_name, center, diag, t, cooldown_s)
        self.last_view: list[dict] = []  # 最近一帧每个目标的状态，供绘图用
        if self.cfg.evidence_dir:
            os.makedirs(self.cfg.evidence_dir, exist_ok=True)

    # -- helpers
    def _has_rider(self, box, persons) -> bool:
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        for p in persons:
            iw = min(box[2], p[2]) - max(box[0], p[0])
            ih = min(box[3], p[3]) - max(box[1], p[1])
            if iw > 0 and ih > 0 and iw * ih / area >= self.cfg.rider_overlap:
                return True
        return False

    def _cooled(self, zone, c, dg, t) -> bool:
        cool = self.cfg.cooldown_s if zone.cooldown_s is None else zone.cooldown_s
        self.alerts = deque(a for a in self.alerts if t - a[3] <= a[4])  # 丢弃已过冷却期的
        for zn, c0, dg0, _, _ in self.alerts:
            if zn == zone.name and math.hypot(c[0] - c0[0], c[1] - c0[1]) < \
                    self.cfg.cooldown_radius_frac * max(dg, dg0):
                return False
        return True

    def _evidence(self, frame, box, zone, tid, t, dwell):
        if frame is None or not self.cfg.evidence_dir:
            return None
        img = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.polylines(img, [zone.polygon.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.putText(img, f"L{zone.level} id{tid} t={t:.0f}s dwell={dwell:.0f}s",
                    (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        path = os.path.join(self.cfg.evidence_dir, f"L{zone.level}_id{tid}_{int(t)}s.jpg")
        return path if _save_jpg(path, img) else None

    def _prune(self):
        alive = self.tracker.tracks
        for tid in [k for k in self.state if k not in alive]:
            del self.state[tid]
        for tid in [k for k in self.rider_since if k not in alive]:
            del self.rider_since[tid]

    # -- main entry
    def update(self, t: float, boxes, classes, persons=None, frame=None):
        """t: 秒（视频时间或时间戳，不是帧号）。boxes: xyxy 列表。persons: 可选人框列表。
        没有检测时也要每帧调用 (boxes=[])，这样丢失轨迹才会按时间过期。"""
        cfg = self.cfg
        ids = self.tracker.update(t, boxes, classes)
        events = []
        self.last_view = []
        for tid, box, cls in zip(ids, boxes, classes):
            ft, c, dg = foot(box), center(box), diag(box)

            rider = persons is not None and self._has_rider(box, persons)
            if rider:
                self.rider_since.setdefault(tid, t)
            else:
                self.rider_since.pop(tid, None)
            in_use = rider and (t - self.rider_since[tid]) >= cfg.rider_min_s

            st = self.state.setdefault(tid, {})
            view = {"id": tid, "box": box, "cls": cls, "zone": None,
                    "dwell": 0.0, "warned": False, "rider": bool(in_use)}
            self.last_view.append(view)
            inside_zones = []
            for z in self.zones:
                s = st.setdefault(z.name, _ZState())
                d = z.signed_dist(ft)
                s.inside = (d >= -cfg.exit_margin_px) if s.inside else (d > cfg.enter_margin_px)
                if s.inside:
                    if s.enter_t is None or t - s.last_inside_t > cfg.gap_s:  # 新的一次停留
                        s.enter_t, s.anchor_c, s.warned = t, c, False
                    s.last_inside_t = t
                    moved = math.hypot(c[0] - s.anchor_c[0], c[1] - s.anchor_c[1]) > cfg.move_frac * dg
                    if in_use or moved:  # 在骑行/推行/调整位置 -> 重新计时，视为新的一次停放
                        s.enter_t, s.anchor_c, s.warned = t, c, False
                    inside_zones.append((z, s))
                elif s.enter_t is not None and t - s.last_inside_t > cfg.gap_s:
                    s.enter_t, s.warned = None, False  # 离开超过 gap_s，停留结束
            if not inside_zones:
                continue
            z, s = min(inside_zones, key=lambda zs: zs[0].level)  # 重叠区域取最严重等级
            dwell = t - s.enter_t
            view.update(zone=z, dwell=dwell, warned=s.warned)
            if dwell >= z.dwell_s and not s.warned:
                s.warned = True
                view["warned"] = True
                if self._cooled(z, c, dg, t):
                    self.alerts.append((z.name, c, dg, t,
                                        self.cfg.cooldown_s if z.cooldown_s is None else z.cooldown_s))
                    events.append({
                        "time": round(t, 2), "zone": z.name, "level": z.level, "track_id": tid,
                        "dwell": round(dwell, 2), "cls": int(cls), "box": [float(v) for v in box],
                        "evidence": self._evidence(frame, box, z, tid, t, dwell),
                    })
        self._prune()
        return events


# ----------------------------------------------------------------- notifier
class WeComNotifier:
    """企业微信群机器人。webhook 从环境变量 WECOM_WEBHOOK 读，不要写进代码/zones.json。
    1/2 级即时发送；3 级和被限频挤出的消息进入汇总，flush_summary() 一次性发。
    网络错误只返回 False，不会让检测主循环崩溃。"""

    def __init__(self, webhook: str | None = None, post=None, max_per_min: int = 18, timeout: float = 5.0):
        self.webhook = webhook or os.environ.get("WECOM_WEBHOOK", "")
        self._post = post
        self.max_per_min, self.timeout = max_per_min, timeout
        self.sent_at: deque = deque()
        self.pending: list[dict] = []

    def _do_post(self, text: str) -> bool:
        payload = {"msgtype": "markdown", "markdown": {"content": text}}
        try:
            post = self._post
            if post is None:
                import requests
                post = requests.post
            r = post(self.webhook, json=payload, timeout=self.timeout)
            return bool(r.json().get("errcode") == 0)
        except Exception:
            return False

    @staticmethod
    def _line(ev: dict) -> str:
        return f"L{ev['level']} {ev['zone']} | id{ev['track_id']} | 已停留 {ev['dwell']:.0f}s | t={ev['time']}"

    def send(self, ev: dict, now: float) -> bool:
        if not self.webhook and self._post is None:
            self.pending.append(ev)
            return False
        if ev["level"] >= 3:
            self.pending.append(ev)
            return True
        while self.sent_at and now - self.sent_at[0] > 60:
            self.sent_at.popleft()
        if len(self.sent_at) >= self.max_per_min:  # 群机器人有频率上限，超了转汇总
            self.pending.append(ev)
            return False
        ok = self._do_post(f"<font color=\"warning\">**违停告警**</font>\n{self._line(ev)}")
        if ok:
            self.sent_at.append(now)
        else:
            self.pending.append(ev)
        return ok

    def flush_summary(self) -> bool:
        if not self.pending:
            return True
        lines = "\n".join(self._line(e) for e in self.pending)
        ok = self._do_post(f"**违停汇总（{len(self.pending)} 条）**\n{lines}")
        if ok:
            self.pending.clear()
        return ok


# --------------------------------------------------------------- evaluation
def evaluate_events(pred, gt, tol_s: float = 5.0, duration_s: float | None = None):
    """事件级评测。
    pred: [{"zone","time"}]  系统告警
    gt:   [{"zone","start","end"}]  人工标注的"真实违停事件"（start=开始停放，end=离开）
    告警时间落在 [start - tol, end + tol] 且 zone 相同 -> TP（一对一匹配）。
    返回 precision / recall / 平均告警延迟(告警时间-start) / 每小时误报数（需 duration_s）。
    """
    gt_used = [False] * len(gt)
    tp, lat = 0, []
    for p in sorted(pred, key=lambda e: e["time"]):
        for i, g in enumerate(gt):
            if not gt_used[i] and g["zone"] == p["zone"] and g["start"] - tol_s <= p["time"] <= g["end"] + tol_s:
                gt_used[i] = True
                tp += 1
                lat.append(p["time"] - g["start"])
                break
    fp = len(pred) - tp
    out = {
        "tp": tp, "fp": fp, "fn": len(gt) - tp,
        "precision": tp / len(pred) if pred else None,
        "recall": tp / len(gt) if gt else None,
        "mean_latency_s": sum(lat) / len(lat) if lat else None,
    }
    if duration_s:
        out["false_alarms_per_hour"] = fp / (duration_s / 3600.0)
    return out


# ---------------------------------------------------------------- self-test
def _run(judge, dur_s, box_fn, fps=5, dropout=(), persons_fn=None):
    events = []
    for k in range(int(dur_s * fps)):
        t = k / fps
        missing = any(a <= t < b for a, b in dropout)
        boxes = [] if missing else [box_fn(t)]
        persons = persons_fn(t) if persons_fn else None
        events += judge.update(t, boxes, [0] * len(boxes), persons=persons)
    return events


def _selftest():
    fire = Zone("fire_lane", 1, 10.0, np.array([[100, 100], [500, 100], [500, 400], [100, 400]], np.float32))
    walk = Zone("walkway", 3, 180.0, np.array([[0, 0], [640, 0], [640, 480], [0, 480]], np.float32))
    rng = np.random.default_rng(0)
    still = lambda t: (250, 200, 330, 300)
    new = lambda zs=(fire,), **kw: ParkingJudge(list(zs), JudgeConfig(**kw))

    # A. 脚点在边界附近抖动（离边界 8px，噪声 ±6px）：滞回后应只告警一次，且约 10s 时触发
    jit = lambda t: (250, 292 + rng.uniform(-6, 6), 330, 392 + rng.uniform(-6, 6))
    ev = _run(new(), 40, jit)
    assert len(ev) == 1 and 9.8 <= ev[0]["time"] <= 11.5, ev
    print("A 边界抖动: 1 次告警, t=%.1fs  OK" % ev[0]["time"])

    # B. 2 秒遮挡（漏检）：计时不中断，仍在约 10s 告警
    ev = _run(new(), 30, still, dropout=[(8.0, 10.0)])
    assert len(ev) == 1 and 9.8 <= ev[0]["time"] <= 10.6 and ev[0]["track_id"] == 1, ev
    print("B 2s 遮挡: 仍为同一 id, t=%.1fs  OK" % ev[0]["time"])

    # C. 缓慢推行穿过禁停区（30px/s，在区内 >10s）：位移重置计时，不应告警
    ev = _run(new(), 14, lambda t: (110 + 30 * t, 200, 190 + 30 * t, 300))
    assert ev == [], ev
    print("C 推行穿过: 0 次告警  OK")

    # D. 有人骑着停了 30s 后离开：人在时不告警，人走后再停 10s 告警
    person = lambda t: [(255, 120, 325, 300)] if t < 30 else []
    ev = _run(new(), 60, still, persons_fn=person)
    assert len(ev) == 1 and 39.5 <= ev[0]["time"] <= 41.5, ev
    print("D 骑手离开: t=%.1fs 告警  OK" % ev[0]["time"])

    # E. 5s 丢失 > gap_s 导致新 id：冷却按"区域+位置"抑制重复；冷却很短时则会重复告警
    ev = _run(new(), 60, still, dropout=[(15.0, 20.0)])
    assert len(ev) == 1, ev
    ev2 = _run(new(cooldown_s=5.0), 60, still, dropout=[(15.0, 20.0)])
    assert len(ev2) == 2 and ev2[0]["track_id"] != ev2[1]["track_id"], ev2
    print("E 冷却: 120s 冷却=1 次, 5s 冷却=2 次  OK")

    # F. 重叠区域取最严重等级
    ev = _run(new(zs=(walk, fire)), 20, still)
    assert len(ev) == 1 and ev[0]["level"] == 1 and ev[0]["zone"] == "fire_lane", ev
    print("F 重叠区域: 取 1 级  OK")

    # G. 跟踪与检测顺序无关
    tk = Tracker()
    a, b = (0, 0, 100, 100), (300, 300, 400, 400)
    assert tk.update(0, [a, b], [0, 0]) == [1, 2]
    assert tk.update(1, [b, a], [0, 0]) == [2, 1]  # 检测顺序反过来，id 仍跟着物体走
    # 一个检测同时与两条轨迹重叠时，应归给 IoU 更大的那条
    tk2 = Tracker()
    tk2.update(0, [(0, 0, 100, 100), (60, 0, 160, 100)], [0, 0])
    assert tk2.update(1, [(58, 0, 158, 100)], [0]) == [2]
    print("G 跟踪与检测顺序无关, 冲突时取最大 IoU  OK")

    # H. 推送：限频 / 3 级汇总 / 失败不抛异常
    calls = []

    class R:
        def json(self): return {"errcode": 0}

    n = WeComNotifier(webhook="http://x", post=lambda url, json, timeout: (calls.append(json), R())[1], max_per_min=18)
    e2 = {"level": 2, "zone": "stair", "track_id": 1, "dwell": 60, "time": 1}
    for i in range(25):
        n.send(e2, now=1000 + i)
    n.send({**e2, "level": 3}, now=1030)
    assert len(calls) == 18 and len(n.pending) == 8, (len(calls), len(n.pending))
    assert n.flush_summary() and len(calls) == 19 and not n.pending
    bad = WeComNotifier(webhook="http://x", post=lambda *a, **k: 1 / 0)
    assert bad.send(e2, now=0) is False  # 不抛异常
    print("H 推送: 18 条/分钟限频, 其余汇总, 异常被吞  OK")

    # I. 事件级评测
    m = evaluate_events([{"zone": "fire_lane", "time": 112}, {"zone": "fire_lane", "time": 900}],
                        [{"zone": "fire_lane", "start": 100, "end": 400}], duration_s=3600)
    assert m["tp"] == 1 and m["fp"] == 1 and m["recall"] == 1.0 and m["precision"] == 0.5
    assert m["mean_latency_s"] == 12 and m["false_alarms_per_hour"] == 1.0
    print("I 事件级评测  OK")

    # J. load_zones 配置错误要大声报
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"zones": [{"name": "x", "level": 1, "dwell_s": 10, "polygon": [[0, 0], [1, 1]]}]}, f)
    try:
        load_zones(f.name, 640, 480)
        raise AssertionError("应该报错")
    except ValueError:
        print("J 无效 zones.json 会报错而不是静默跳过  OK")
    os.unlink(f.name)

    # K. 证据截图（含中文目录名）
    with tempfile.TemporaryDirectory() as d:
        ed = os.path.join(d, "证据")
        j = ParkingJudge([fire], JudgeConfig(evidence_dir=ed))
        frame = np.zeros((480, 640, 3), np.uint8)
        evs = []
        for k in range(60):
            evs += j.update(k / 5, [(250, 200, 330, 300)], [0], frame=frame)
        assert evs and evs[0]["evidence"] and os.path.getsize(evs[0]["evidence"]) > 0
        print("K 证据截图写入中文路径  OK")
    # L. 项目实际的 zones.json 格式（归一化 points / dwell / cooldown / color）
    cfg = {"zones": [{"name": "消防通道", "level": 1, "dwell": 10, "cooldown": 300, "color": [0, 0, 255],
                      "points": [[0.0, 0.45], [0.45, 0.38], [0.50, 1.0], [0.0, 1.0]]}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8-sig") as f:
        json.dump(cfg, f, ensure_ascii=False)
    zs = load_zones(f.name, 960, 540)
    os.unlink(f.name)
    assert zs[0].cooldown_s == 300 and zs[0].color == (0, 0, 255) and abs(zs[0].polygon[0][1] - 243) < 1e-3
    assert zs[0].signed_dist((215, 470)) > 0 > zs[0].signed_dist((800, 100))
    # 区域自带的冷却时间优先于全局默认：冷却 300s，第二次(30s)被抑制
    j = ParkingJudge(zs, JudgeConfig(cooldown_s=1.0))
    ev = _run(j, 60, lambda t: (100, 200, 330, 470), dropout=[(15.0, 20.0)])
    assert len(ev) == 1, ev
    print("L 项目真实 zones.json 格式 + 区域自带冷却  OK")
    print("\nALL PASSED")


if __name__ == "__main__":
    _selftest()
