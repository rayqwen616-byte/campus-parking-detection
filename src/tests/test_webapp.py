# -*- coding: utf-8 -*-
"""webapp 后端单测：区域校验、编解码、脚点判定（不需要加载模型）"""
import json
import sys
from pathlib import Path

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
    url = webapp.encode_jpeg(img)
    assert url.startswith("data:image/jpeg;base64,")
    back = webapp.decode_data_url(url)
    assert back.shape[0] == 120 and back.shape[1] == 160

    # 11 非法 dataURL
    assert expect_error(webapp.decode_data_url, "http://example.com/a.jpg")

    print("WEBAPP PASSED")


if __name__ == "__main__":
    main()
