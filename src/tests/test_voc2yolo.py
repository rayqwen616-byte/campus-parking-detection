# -*- coding: utf-8 -*-
"""voc2yolo.py 配对逻辑测试（v8 版）：合成一个小 VOC 数据集，覆盖 6 种情况。

覆盖：
    1 文件名直接配上
    2 靠 XML 内 <filename> 救回
    3 尺寸不符 -> 拒绝
    4 找不到图片 -> 记录
    5 两个 XML 认领同一张图 -> 去重
    6 宽高互换（EXIF 旋转）-> 允许
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = Path(tempfile.mkdtemp(prefix="voc_test_"))
R = T / "mydata"
(R / "Annotations").mkdir(parents=True)
(R / "JPEGImages").mkdir()


def img(name, w, h):
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    buf.tofile(str(R / "JPEGImages" / name))


def xml(name, fn, w, h):
    (R / "Annotations" / name).write_text(
        "<annotation><filename>%s</filename>"
        "<size><width>%d</width><height>%d</height><depth>3</depth></size>"
        "<object><name>motorcycle</name><bndbox>"
        "<xmin>10</xmin><ymin>10</ymin><xmax>80</xmax><ymax>90</ymax>"
        "</bndbox></object></annotation>" % (fn, w, h),
        encoding="utf-8")


img("a.jpg", 200, 100); xml("a.xml", "a.jpg", 200, 100)                  # 1 按名配对
img("校园_001.jpg", 300, 200); xml("663.xml", "校园_001.jpg", 300, 200)   # 2 靠 <filename> 救回
img("b.jpg", 400, 300); xml("700.xml", "b.jpg", 999, 999)                # 3 尺寸不符
xml("701.xml", "nothing.jpg", 100, 100)                                  # 4 找不到图
xml("702.xml", "校园_001.jpg", 300, 200)                                 # 5 重复认领
img("c.jpg", 640, 480); xml("703.xml", "c.jpg", 480, 640)                # 6 宽高互换

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
p = subprocess.run([sys.executable, str(ROOT / "voc2yolo.py"),
                    "--src", str(R), "--out", str(T / "out"), "--copy"],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace", env=env)
assert p.returncode == 0, (p.stdout or "") + (p.stderr or "")

flat = " ".join(p.stdout.split())
assert "有效 3 张" in flat, flat
assert "按相对路径 1" in flat, flat
assert "按<filename>回退救回 2" in flat, flat
assert "尺寸不符 1" in flat, flat
assert "重复 1" in flat, flat
assert "缺图 1" in flat, flat

txt = (T / "out" / "unpaired.txt").read_text(encoding="utf-8")
assert "700.xml" in txt        # 尺寸不符
assert "701.xml" in txt        # 找不到图
assert "702.xml" in txt        # 重复
assert "663.xml" not in txt    # 被救回，不该出现在未配对清单

labels = {q.stem for q in (T / "out" / "labels").rglob("*.txt")}
assert labels == {"a", "校园_001", "c"}, labels

print("VOC2YOLO PASSED")
