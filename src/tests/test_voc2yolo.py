# -*- coding: utf-8 -*-
"""voc2yolo.py 配对逻辑测试：合成一个小 VOC 数据集，覆盖 6 种情况。"""
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
        f"""<annotation><filename>{fn}</filename><size><width>{w}</width><height>{h}</height><depth>3</depth></size>
<object><name>motorcycle</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>80</xmax><ymax>90</ymax></bndbox></object></annotation>""",
        encoding="utf-8")


img("a.jpg", 200, 100); xml("a.xml", "a.jpg", 200, 100)                 # 1 文件名直接配上
img("校园_001.jpg", 300, 200); xml("663.xml", "校园_001.jpg", 300, 200)   # 2 靠 <filename> 救回
img("b.jpg", 400, 300); xml("700.xml", "b.jpg", 999, 999)               # 3 尺寸不符 -> 拒绝
xml("701.xml", "nothing.jpg", 100, 100)                                   # 4 找不到图片
xml("702.xml", "校园_001.jpg", 300, 200)                                  # 5 与 663 指向同一张图 -> 重复
img("c.jpg", 640, 480); xml("703.xml", "c.jpg", 480, 640)               # 6 宽高互换（EXIF 旋转）-> 允许

p = subprocess.run([sys.executable, str(ROOT / "voc2yolo.py"), "--src", str(R), "--out", str(T / "out"), "--copy"],
                   capture_output=True, text=True, encoding="utf-8")
assert p.returncode == 0, p.stderr
assert "有效 3 张（按文件名 1，按<filename>回退救回 2）" in p.stdout, p.stdout
assert "尺寸不符 1，重复 1" in p.stdout, p.stdout
txt = (T / "out" / "unpaired.txt").read_text(encoding="utf-8")
assert "700.xml" in txt and "701.xml" in txt and "702.xml" in txt and "663.xml" not in txt, txt
labels = {q.stem for q in (T / "out" / "labels").rglob("*.txt")}
assert labels == {"a", "校园_001", "c"}, labels
print("VOC2YOLO PASSED")
