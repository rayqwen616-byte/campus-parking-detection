# -*- coding: utf-8 -*-
"""
zones2yolo_seg.py —— 把 labelme 标注的多边形，转成 YOLO 分割数据集

准备：
    1. 用 labelme 打开图片，沿边界画多边形，类别填 forbidden / passage / legal
       （legal = 白线车位等合法停车区，用于和禁停区形成对比）
    2. 保存后会得到同名 .json 文件

目录要求：
    data/
    ├── 001.jpg
    ├── 001.json
    ├── 002.jpg
    └── 002.json

用法：
    python zones2yolo_seg.py --src D:/zones_raw --out D:/zones_ds

输出：
    zones_ds/
    ├── images/train, images/val
    ├── labels/train, labels/val
    └── data.yaml
"""
import argparse
import json
import random
import shutil
from pathlib import Path

CLASSES = ["forbidden", "passage", "legal"]   # 0=禁停区 1=通道 2=合法车位（白线内）
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def convert_one(json_path: Path, img_w, img_h):
    """
    labelme JSON → YOLO 分割标签行
    格式：class x1 y1 x2 y2 ...（全部归一化到 0~1）
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        if label not in CLASSES:
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        cls_id = CLASSES.index(label)
        coords = []
        for x, y in pts:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="labelme 标注所在目录")
    ap.add_argument("--out", required=True, help="输出的数据集目录")
    ap.add_argument("--val", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    random.seed(args.seed)

    # ---------- 收集成对的文件 ----------
    pairs = []
    skipped = 0
    for js in sorted(src.glob("*.json")):
        img = None
        for ext in IMG_EXT:
            p = src / (js.stem + ext)
            if p.exists():
                img = p
                break
        if img is None:
            skipped += 1
            continue
        pairs.append((img, js))

    print(f"[scan] 找到 {len(pairs)} 组图片+标注，跳过 {skipped} 组（缺图片）")
    if not pairs:
        raise SystemExit("没有可用数据，请检查目录")

    # ---------- 从图片读取真实尺寸 ----------
    import cv2
    random.shuffle(pairs)
    n_val = max(int(len(pairs) * args.val), 1)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        n_poly, n_empty = 0, 0
        for img, js in items:
            frame = cv2.imread(str(img))
            if frame is None:
                print(f"  跳过无法读取的图片：{img.name}")
                continue
            h, w = frame.shape[:2]
            lines = convert_one(js, w, h)

            if not lines:                      # 没有任何有效多边形
                n_empty += 1
                continue

            shutil.copy2(img, out / "images" / split / img.name)
            (out / "labels" / split / (img.stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            n_poly += len(lines)

        print(f"  [{split}] {len(items)} 组 -> 有效 {n_poly} 个多边形，跳过空标注 {n_empty} 张")

    # ---------- 写 data.yaml ----------
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")
    print(f"\n[done] {yaml_path}")
    print("下一步：python train_zones.py --data " + str(yaml_path))


if __name__ == "__main__":
    main()
