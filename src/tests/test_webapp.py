# -*- coding: utf-8 -*-
"""webapp 后端单测：区域校验、编解码、脚点判定（不需要加载模型）"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import webapp  # noqa: E402


def expect_error(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def main():
    W, H = 1000, 800

    # 1 合法区域
    zones = webapp.zones_from_payload(
        [{"name": "测试", "level": 1, "points": [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0]]}], W, H)
    assert len(zones) == 1 and zones[0].name == "测试" and zones[0].level == 1

    # 2 顶点不足
    assert expect_error(webapp.zones_from_payload,
                        [{"name": "x", "points": [[0, 0], [1, 1]]}], W, H)

    # 3 坐标越界
    assert expect_error(webapp.zones_from_payload,
                        [{"name": "x", "points": [[0, 0], [1, 1], [2, 0.5]]}], W, H)

    # 4 等级非法
    assert expect_error(webapp.zones_from_payload,
                        [{"name": "x", "level": 9,
                          "points": [[0, 0], [1, 1], [0.5, 0.5]]}], W, H)

    # 5 颜色非法
    assert expect_error(webapp.zones_from_payload,
                        [{"name": "x", "color": [0, 0, 999],
                          "points": [[0, 0], [1, 1], [0.5, 0.5]]}], W, H)

    # 6 空列表合法
    assert webapp.zones_from_payload([], W, H) == []

    # 7 脚点在区域内
    z = webapp.zones_from_payload(
        [{"name": "内", "level": 2, "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}], W, H)[0]
    assert webapp.find_zone([z], (500, 400)) is z

    # 8 脚点在区域外
    assert webapp.find_zone([z], (500, -10)) is None

    # 9 多区域取等级最小
    z1 = webapp.zones_from_payload(
        [{"name": "L3", "level": 3, "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}], W, H)[0]
    z2 = webapp.zones_from_payload(
        [{"name": "L1", "level": 1, "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}], W, H)[0]
    assert webapp.find_zone([z1, z2], (500, 400)).name == "L1"

    # 10 JPEG 编解码往返
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    raw = webapp.encode_jpeg(img)
    url = "data:image/jpeg;base64," + raw
    assert raw.startswith("/9j/"), "应为纯 base64（JPEG 以 /9j/ 开头）"
    back = webapp.decode_data_url(url)
    assert back.shape[0] == 120 and back.shape[1] == 160

    # 11 非法 dataURL
    assert expect_error(webapp.decode_data_url, "http://example.com/a.jpg")

    # 12 网页默认只保留车辆白名单，过滤人、汽车等通用模型类别
    boxes = SimpleNamespace(
        xyxy=Tensor([[0, 0, 30, 50], [35, 0, 65, 50], [0, 55, 30, 99], [35, 55, 65, 99]]),
        cls=Tensor([0, 1, 2, 3]),
        conf=Tensor([0.9, 0.8, 0.95, 0.85]),
    )
    result = SimpleNamespace(names={0: "ebike", 1: "bicycle", 2: "person", 3: "car"}, boxes=boxes)
    original_model = webapp.State.model
    webapp.State.model = lambda *args, **kwargs: [result]
    try:
        dets, summary = webapp.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0.25, [])
    finally:
        webapp.State.model = original_model
    assert [d["model_label"] for d in dets] == ["ebike", "bicycle"], dets
    assert summary == {"total": 2, "inside": 0, "outside": 2}, summary

    # 13 自动区域只接受 forbidden，passage 不得变成 L3 告警区
    masks = np.zeros((2, 100, 100), dtype=np.float32)
    masks[0, 10:60, 10:60] = 1
    masks[1, 60:95, 60:95] = 1
    seg_result = SimpleNamespace(
        names={0: "forbidden", 1: "passage"},
        boxes=SimpleNamespace(cls=Tensor([0, 1])),
        masks=SimpleNamespace(data=Tensor(masks)),
    )
    auto_zones = webapp.zones_from_segmentation(seg_result, 100, 100)
    assert len(auto_zones) == 1 and auto_zones[0]["level"] == 1, auto_zones
    xs = [p[0] for p in auto_zones[0]["points"]]
    assert max(xs) <= 0.6, auto_zones

    print("WEBAPP PASSED (13 tests)")


if __name__ == "__main__":
    main()
