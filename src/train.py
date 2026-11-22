# -*- coding: utf-8 -*-
"""
校园非机动车违停检测 —— 训练 / 验证 / 导出 一条龙脚本。

用法（矩池云 RTX 4090 实例上）：
    python train.py --data /mnt/data/campus_bike/data.yaml --stage all

分步执行：
    python train.py --stage check      # 只检查数据集
    python train.py --stage train      # 训练
    python train.py --stage val        # 在 test 集上验证
    python train.py --stage export     # 导出 onnx + ncnn
    python train.py --resume           # 断点续训

依赖：pip install -U ultralytics
"""
import argparse
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset(data_yaml: Path) -> None:
    """轻量检查：图片/标签是否一一对应，标签格式是否合法。"""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    nc = len(cfg["names"])
    print(f"[check] root = {root}, classes = {nc}")
    ok = True
    for split in ("train", "val", "test"):
        img_dir = root / cfg[split]
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            print(f"  [{split}] 缺少图片目录: {img_dir}")
            ok = False
            continue
        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
        if not imgs:
            print(f"  [{split}] 没有图片")
            ok = False
            continue
        missing, bad, boxes = 0, 0, 0
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                missing += 1
                continue
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= cid < nc) or any(v < 0 or v > 1 for v in vals):
                    bad += 1
                else:
                    boxes += 1
        print(f"  [{split}] 图片 {len(imgs)}，缺失标签 {missing}，非法行 {bad}，标注框 {boxes}")
        if missing or bad:
            ok = False
    print("[check] 通过" if ok else "[check] 有问题，先修数据再训练")
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/data/campus_bike/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--stage", default="all",
                    choices=["check", "train", "val", "export", "all"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="test", help="验证用哪个 split，没有 test 会自动退回 val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="/mnt/runs")
    ap.add_argument("--name", default="yolo11n_960")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        sys.exit(f"找不到 {data_yaml}，先按 data.yaml 模板改路径")

    from ultralytics import YOLO

    if args.stage in ("check", "all"):
        check_dataset(data_yaml)

    best = Path(args.project) / args.name / "weights" / "best.pt"

    if args.stage in ("train", "all") or args.resume:
        if args.resume:
            last = Path(args.project) / args.name / "weights" / "last.pt"
            model = YOLO(str(last))
            print(f"[train] 断点续训: {last}")
            model.train(resume=True)
        else:
            model = YOLO(args.model)
            print(f"[train] {args.model} -> imgsz={args.imgsz} batch={args.batch}")
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                optimizer="SGD",
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                cos_lr=True,
                close_mosaic=20,
                patience=50,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=5,
                translate=0.1,
                scale=0.5,
                mosaic=1.0,
                mixup=0.1,
                fliplr=0.5,
                erasing=0.4,
                workers=args.workers,
                amp=True,
                cache="ram",
                device=args.device,
                project=args.project,
                name=args.name,
                exist_ok=True,
            )

    if args.stage in ("val", "all"):
        split = args.split
        if split == "test":
            import yaml
            cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            root = Path(cfg.get("path", data_yaml.parent))
            test_key = cfg.get("test")
            if not test_key or not (root / test_key).exists():
                split = "val"
                print("[val] 数据里没有 test 集，改用 val 评估")
        model = YOLO(str(best))
        m = model.val(data=str(data_yaml), split=split, imgsz=args.imgsz,
                      batch=args.batch, device=args.device)
        print(f"[val:{split}] mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")

    if args.stage in ("export", "all"):
        model = YOLO(str(best))
        model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        model.export(format="ncnn", imgsz=640, half=True)
        print("[export] onnx / ncnn 已生成到 weights 同级目录")


if __name__ == "__main__":
    main()
