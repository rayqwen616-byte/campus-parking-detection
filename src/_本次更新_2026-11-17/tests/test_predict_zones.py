# -*- coding: utf-8 -*-
"""predict_zones 掩码→多边形 单测（不需要模型，用合成掩码）"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict_zones import mask_to_polygons, clean_mask  # noqa: E402


def make_mask(w, h, rects):
    """在 w×h 的掩码上画若干矩形（值为 1）"""
    m = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in rects:
        m[y1:y2, x1:x2] = 1.0
    return m


def main():
    W, H = 400, 300

    # 1 单个矩形 → 4~8 个顶点的多边形，坐标在 0~1
    polys = mask_to_polygons(make_mask(W, H, [(100, 100, 300, 250)]), W, H)
    assert len(polys) == 1, polys
    pts = polys[0]
    assert 3 <= len(pts) <= 10, len(pts)
    for x, y in pts:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    # 2 面积占比正确：矩形 200×150 / (400×300) = 25%
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    assert 0.20 < area < 0.30, area

    # 3 太小的区域被过滤（min_area=0.05，小矩形仅 4%）
    small = make_mask(W, H, [(10, 10, 50, 40)])
    assert mask_to_polygons(small, W, H, min_area_ratio=0.05) == []
    # 放宽阈值就能出来
    assert len(mask_to_polygons(small, W, H, min_area_ratio=0.005)) == 1

    # 4 all_contours 提取多个
    two = make_mask(W, H, [(10, 10, 120, 120), (250, 150, 390, 290)])
    assert len(mask_to_polygons(two, W, H, all_contours=True, min_area_ratio=0.01)) == 2
    assert len(mask_to_polygons(two, W, H, all_contours=False)) == 1

    # 5 空掩码
    assert mask_to_polygons(np.zeros((H, W), dtype=np.float32), W, H) == []

    # 6 闭运算补洞：中间挖空仍只有一个轮廓
    holed = make_mask(W, H, [(100, 100, 300, 250)])
    holed[150:200, 180:220] = 0
    assert len(mask_to_polygons(holed, W, H, morph=15)) == 1

    # 7 clean_mask 输出 0/255
    m = clean_mask(make_mask(W, H, [(10, 10, 100, 100)]), 5)
    assert set(np.unique(m)).issubset({0, 255})

    print("PREDICT_ZONES PASSED")


if __name__ == "__main__":
    main()
