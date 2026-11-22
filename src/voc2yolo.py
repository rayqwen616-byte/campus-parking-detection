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


def build_index(img_dir: Path):
    """图片索引：按完整文件名 / 按 stem（均不区分大小写），含子目录"""
    by_name, by_stem = {}, {}
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            by_name.setdefault(p.name.lower(), p)
            by_stem.setdefault(p.stem.lower(), p)
    return by_name, by_stem


def xml_hints(xml_path: Path):
    """从 XML 里取 <filename>/<path> 指向的图片名，以及 <size> 的宽高"""
    root = ET.parse(xml_path).getroot()
    hints = []
    for tag in ("filename", "path"):
        v = (root.findtext(tag) or "").strip()
        if v:
            hints.append(Path(v.replace("\\", "/")).name)
    size = root.find("size")
    wh = None
    if size is not None:
        try:
            wh = (int(size.find("width").text), int(size.find("height").text))
        except (AttributeError, TypeError, ValueError):
            pass
    return hints, wh


def size_ok(img: Path, wh) -> bool:
    """回退配对的安全校验：图片尺寸必须与 XML 的 <size> 一致（允许 EXIF 旋转造成的宽高互换）"""
    if wh is None:
        return False
    from PIL import Image
    with Image.open(img) as im:
        iw, ih = im.size
    return (iw, ih) == wh or (ih, iw) == wh


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

    by_name, by_stem = build_index(img_dir)
    pairs, no_box = [], 0
    n_stem = n_fn = n_size_bad = n_dup = 0
    unpaired = []
    used = set()
    for xml in sorted(ann_dir.rglob("*.xml")):
        img, how = by_stem.get(xml.stem.lower()), "stem"
        hints, wh = [], None
        if img is None:  # 回退：按 XML 里的 <filename>/<path> 配对，并校验尺寸
            hints, wh = xml_hints(xml)
            for h in hints:
                cand = by_name.get(h.lower()) or by_stem.get(Path(h).stem.lower())
                if cand is not None:
                    img, how = cand, "filename"
                    break
            if img is not None and not size_ok(img, wh):
                n_size_bad += 1
                unpaired.append(f"{xml.name}\t<filename> 指向 {img.name}，但尺寸与 XML 不符，已拒绝")
                continue
        if img is None:
            unpaired.append(f"{xml.name}\t找不到对应图片")
            continue
        if img in used:  # 两个 XML 指向同一张图，只保留第一个，避免同图两套标注
            n_dup += 1
            unpaired.append(f"{xml.name}\t与前一个 XML 指向同一张图 {img.name}")
            continue
        boxes = convert_one(xml, img)
        if not boxes:
            no_box += 1
            continue
        used.add(img)
        pairs.append((img, boxes))
        n_stem += how == "stem"
        n_fn += how == "filename"

    out.mkdir(parents=True, exist_ok=True)
    (out / "unpaired.txt").write_text("\n".join(unpaired) + ("\n" if unpaired else ""), encoding="utf-8")
    print(f"[scan] 有效 {len(pairs)} 张（按文件名 {n_stem}，按<filename>回退救回 {n_fn}）；"
          f"无目标 {no_box} 张；仍未配上 {len(unpaired)} 个（尺寸不符 {n_size_bad}，重复 {n_dup}）")
    if unpaired:
        print(f"       未配上的清单：{out / 'unpaired.txt'}")
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
