# -*- coding: utf-8 -*-
"""eval_zones 的核心几何/解析函数测试（不需要模型）"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_zones import iou_mask, load_gt, poly_mask  # noqa: E402


def main():
    W, H = 100, 100

    # 1 完全相同的多边形 -> IoU 1.0
    sq = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    a = poly_mask(sq, W, H)
    b = poly_mask(sq, W, H)
    assert abs(iou_mask(a, b) - 1.0) < 1e-6

    # 2 完全不相交 -> 0
    far = [[0.7, 0.7], [0.9, 0.7], [0.9, 0.9], [0.7, 0.9]]
    assert iou_mask(a, poly_mask(far, W, H)) == 0.0

    # 3 小矩形完全落在大的里面 -> IoU = 0.08/0.16 = 0.5
    shifted = [[0.10, 0.10], [0.30, 0.10], [0.30, 0.50], [0.10, 0.50]]
    c = poly_mask(shifted, W, H)
    iou = iou_mask(a, c)
    assert 0.45 < iou < 0.55, iou

    # 4 空掩码不除零
    empty = np.zeros((H, W), bool)
    assert iou_mask(empty, empty) == 0.0

    # 5 load_gt 解析 labelme
    tmp = Path(tempfile.mkdtemp(prefix="evalz_"))
    jp = tmp / "x.json"
    jp.write_text(json.dumps({
        "shapes": [
            {"label": "forbidden", "points": [[0, 0], [50, 0], [50, 50]]},
            {"label": "Passage", "points": [[60, 60], [90, 60], [90, 90]]},
            {"label": "其他", "points": [[1, 1], [2, 2], [3, 3]]},
            {"label": "forbidden", "points": [[1, 1], [2, 2]]},   # 点数不足
        ]
    }), encoding="utf-8")

    gt = load_gt(jp, W, H)
    assert set(gt.keys()) == {"forbidden", "passage"}, gt.keys()
    assert len(gt["forbidden"]) == 1          # 点数不足的被丢弃
    assert len(gt["passage"]) == 1
    # 坐标归一化正确
    assert abs(gt["forbidden"][0][1][0] - 0.5) < 1e-6

    print("EVAL_ZONES PASSED")


if __name__ == "__main__":
    main()
