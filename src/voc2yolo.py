# -*- coding: utf-8 -*-
"""
voc2yolo.py —— VOC 转 YOLO（v8 合并版配对策略）

配对顺序（三级）：
    1) 相对路径 + 零填充归一化匹配：train/601.xml  ↔  train/000601.jpg
    2) 回退：读取 XML 内部的 <filename> 字段，按文件名再找一次
    3) 尺寸校验：XML 声明的宽高必须与图片实际尺寸一致，否则判为错配、跳过

为什么这么做：
    · 该数据集里存在 601.xml ↔ 000601.jpg 这种零填充差异（第 1 级解决）
    · 另有 663.xml ↔ 电动车1181.jpg 这种完全换名的（第 1 级无解，第 2 级靠 <filename> 救回）
    · 尺寸校验能挡住"名字对上但内容错位"的假配对（第 3 级）

用法：
    python voc2yolo.py --src D:/datasets/mydata --out D:/datasets/ebike

输出：
    <out>/images/{train,val,test}、<out>/labels/{train,val,test}
    <out>/data.yaml
    <out>/unpaired.txt   未能配对的清单及原因
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

CLASS_MAP = {
    "motorcycle": 0, "motorcycle ": 0,
    "bicycle": 1,
    "cycle": 1,
    "ebike": 0,
}
NAMES = ["ebike", "bicycle"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def imread_u(path):
    """兼容 Windows 中文路径的图片读取（cv2.imread 对中文路径会静默失败）"""
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_dirs(src: Path):
    ann = img = None
    for p in src.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name == "annotations":
            ann = p
        elif name in ("jpegimages", "images"):
            img = p
    if not ann or not img:
        raise SystemExit(f"{src} 下找不到 Annotations / JPEGImages")
    return ann, img


def _norm(part: str) -> str:
    """把纯数字的文件名主干去掉前导零：000601 -> 601"""
    return str(int(part)) if part.isdigit() else part


def build_indexes(img_dir: Path):
    """建立两张索引：相对路径(零填充归一化) → 图片；文件名 → 图片"""
    by_rel, by_name = {}, {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        rel = p.relative_to(img_dir).with_suffix("")
        by_rel.setdefault("/".join(_norm(x) for x in rel.parts), p)
        by_name.setdefault(p.name, p)
    return by_rel, by_name


def read_xml(xml_path: Path):
    """返回 (W, H, filename, [(cls, cx, cy, w, h)])"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(float(size.find("width").text or 0)) if size is not None else 0
    H = int(float(size.find("height").text or 0)) if size is not None else 0
    fname = (root.findtext("filename") or "").strip()

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_MAP:
            continue
        b = obj.find("bndbox")
        if b is None:
            continue
        x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
        if W <= 0 or H <= 0:
            continue
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        boxes.append((CLASS_MAP[name], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                      (x2 - x1) / W, (y2 - y1) / H))
    return W, H, fname, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="复制图片（默认硬链接）")
    ap.add_argument("--skip-size-check", action="store_true", help="关闭尺寸校验")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    ann_dir, img_dir = find_dirs(src)
    by_rel, by_name = build_indexes(img_dir)
    random.seed(args.seed)

    pairs = []
    used_images = set()
    n_by_name = n_by_meta = 0
    bad_size, no_img, dup, no_box = [], [], [], 0

    for xml in sorted(ann_dir.rglob("*.xml")):
        rel = xml.relative_to(ann_dir).with_suffix("")
        key = "/".join(_norm(x) for x in rel.parts)

        # 第 1 级：相对路径（零填充归一化）
        img = by_rel.get(key)
        hit = "name"
        # 第 2 级：<filename> 回退
        if img is None:
            W0, H0, fname, _ = read_xml(xml)
            if fname and fname in by_name:
                img = by_name[fname]
                hit = "meta"
        if img is None:
            no_img.append(xml.name)
            continue

        W, H, fname, boxes = read_xml(xml)

        # 第 3 级：尺寸校验
        if not args.skip_size_check:
            im = imread_u(img)
            if im is None:
                no_img.append(xml.name)
                continue
            h, w = im.shape[:2]
            # 允许宽高互换：手机照片常带 EXIF 旋转，XML 记录的可能是旋转前的尺寸
            if (W, H) != (w, h) and (W, H) != (h, w):
                bad_size.append(f"{xml.name} -> {img.name}  XML:{W}x{H}  图片:{w}x{h}")
                continue

        # 重复：同一张图被两个 XML 认领，只保留先出现的那个
        key_img = str(img.resolve())
        if key_img in used_images:
            dup.append(f"{xml.name} -> {img.name}")
            continue
        used_images.add(key_img)

        if not boxes:
            no_box += 1
            continue

        pairs.append((img, boxes))
        if hit == "name":
            n_by_name += 1
        else:
            n_by_meta += 1

    print(f"[scan] 有效 {len(pairs)} 张（按相对路径 {n_by_name}，"
          f"按<filename>回退救回 {n_by_meta}）；"
          f"尺寸不符 {len(bad_size)}；重复 {len(dup)}；"
          f"缺图 {len(no_img)}；无目标 {no_box}")

    out.mkdir(parents=True, exist_ok=True)
    report = out / "unpaired.txt"
    lines = ["# 未能配对的标注清单", "",
             f"## 尺寸不符（{len(bad_size)}）"] + bad_size
    lines += ["", f"## 重复认领同一张图（{len(dup)}）"] + dup
    lines += ["", f"## 找不到图片（{len(no_img)}）"] + no_img
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if bad_size or no_img or dup:
        print(f"       明细见：{report}")

    if not pairs:
        raise SystemExit("没有可用数据")

    random.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    splits = {"test": pairs[:n_test],
              "val": pairs[n_test:n_test + n_val],
              "train": pairs[n_test + n_val:]}

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        nb = 0
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
            (out / "labels" / split / (img.stem + ".txt")).write_text(txt + "\n",
                                                                     encoding="utf-8")
            nb += len(boxes)
        print(f"  [{split}] {len(items)} 张 / {nb} 个框")

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnames:\n" +
        "".join(f"  {i}: {nm}\n" for i, nm in enumerate(NAMES)),
        encoding="utf-8")
    print(f"\n[done] {out}/data.yaml")


if __name__ == "__main__":
    main()
