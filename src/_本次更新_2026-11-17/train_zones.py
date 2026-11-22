# -*- coding: utf-8 -*-
"""
train_zones.py —— 训练禁停区分割模型

用法（有网可下载预训练权重时，效果更好）：
    python train_zones.py --data D:/zones_ds/data.yaml

用法（无网，从零训练）：
    python train_zones.py --data D:/zones_ds/data.yaml --scratch
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml 路径")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--scratch", action="store_true", help="从零训练（不下载预训练权重）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="0=GPU，cpu=CPU")
    ap.add_argument("--name", default="zones")
    args = ap.parse_args()

    model_path = "yolo11n-seg.yaml" if args.scratch else args.model
    print(f"[模型] {model_path}")
    print(f"[数据] {args.data}")
    print(f"[配置] imgsz={args.imgsz} epochs={args.epochs} batch={args.batch} device={args.device}")

    model = YOLO(model_path)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        amp=False,
        project="runs_zones",
        name=args.name,
    )

    # ultralytics 会自动拼 runs/<task>/ 前缀，取真实保存目录
    try:
        save_dir = Path(model.trainer.save_dir)
    except Exception:
        save_dir = Path("runs/segment/runs_zones") / args.name

    best = (save_dir / "weights" / "best.pt").resolve()
    print("")
    print("=" * 60)
    print("训练完成，最佳模型：")
    print(str(best))
    print("=" * 60)
    print("下一步（复制整行执行）：")
    print(r'D:\yolo_env\Scripts\python.exe predict_zones.py --model "%s" --source D:\zones_raw\A_001.jpg --conf 0.1' % best)


if __name__ == "__main__":
    main()
