# -*- coding: utf-8 -*-
"""
VOC 格式数据集 → YOLO 格式（用于 mydata / motorcycle 这个电动车数据集）

目录要求（解压后的 mydata 文件夹）：
    mydata/
    ├── Annotations/  *.xml
    ├── JPEGImages/   *.jpg / *.png
    └── ImageSets/    可选，忽略

用法：
    python voc2yolo.py --src D:/datasets/mydata --out D:/datasets/ebike

输出：
    ebike/images/{train,val,test}
    ebike/labels/{train,val,test}
    ebike/data.yaml
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 原始类别名 -> 新类别 id
CLASS_MAP = {
    "motorcycle": 0,
    "bicycle": 1,
    "cycle": 1,          # 含义不明的杂类，先并进自行车
}
NAMES = ["ebike", "bicycle"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_dirs(src: Path):
    ann = None
    img = None
    for p in src.rglob("*"):
        if p.is_dir() and p.name.lower() == "annotations":
            ann = p
        if p.is_dir() and p.name.lower() in ("jpegimages", "images"):
            img = p
    if not ann or not img:
        raise SystemExit(f"{src} 下找不到 Annotations / JPEGImages")
    return ann, img


def convert_one(xml_path: Path, img_path: Path):
    """返回 [(cls, cx, cy, w, h)]，坐标已归一化"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    out = []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name not in CLASS_MAP:
            continue
        b = obj.find("bndbox")
        x1 = float(b.find("xmin").text)
        y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text)
        y2 = float(b.find("ymax").text)
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        out.append((CLASS_MAP[name],
                    (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                    (x2 - x1) / W, (y2 - y1) / H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="复制图片（默认硬链接/移动）")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    ann_dir, img_dir = find_dirs(src)
    random.seed(args.seed)

    pairs, no_box, bad = [], 0, 0
    for xml in sorted(ann_dir.rglob("*.xml")):
        stem = xml.stem
        img = None
        for ext in IMG_EXT:
            p = img_dir / (stem + ext)
            if p.exists():
                img = p
                break
        if img is None:
            bad += 1
            continue
        boxes = convert_one(xml, img)
        if not boxes:
            no_box += 1
            continue
        pairs.append((img, boxes))

    print(f"[scan] 有效 {len(pairs)} 张；无目标 {no_box} 张；缺图 {bad} 张")
    random.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    splits = {
        "test": pairs[:n_test],
        "val": pairs[n_test:n_test + n_val],
        "train": pairs[n_test + n_val:],
    }

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        boxes_total = 0
        for img, boxes in items:
            dst = out / "images" / split / img.name
            if not dst.exists():
                if args.copy:
                    shutil.copy2(img, dst)
                else:
                    try:
                        dst.hardlink_to(img)
                    except OSError:
                        shutil.copy2(img, dst)
            txt = "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                            for c, cx, cy, w, h in boxes)
            (out / "labels" / split / (img.stem + ".txt")).write_text(
                txt + "\n", encoding="utf-8")
            boxes_total += len(boxes)
        print(f"  [{split}] {len(items)} 张 / {boxes_total} 个框")

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8")
    print(f"\n[done] {out}/data.yaml")


if __name__ == "__main__":
    main()
