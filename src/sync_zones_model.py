# -*- coding: utf-8 -*-
"""
sync_zones_model.py —— 把最新训练出的区域模型复制成 D:/models/zones.pt

训练完老是记不住 best.pt 在哪，跑这一个脚本就自动找到最新的一份并复制过去，
webapp / predict_zones / demo_v8 都按 D:/models/zones.pt 这个固定路径找。

用法：
    python sync_zones_model.py
    python sync_zones_model.py --dry-run     只看会用哪个文件
"""
import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
TARGET = Path("D:/models/zones.pt")


def find_latest():
    cands = list(BASE.glob("runs/segment/runs_zones/*/weights/best.pt"))
    cands += list(BASE.glob("runs_zones/*/weights/best.pt"))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", default=str(TARGET))
    args = ap.parse_args()

    src = find_latest()
    if src is None:
        raise SystemExit("没找到任何 best.pt，先跑 train_zones.py")

    size = src.stat().st_size / 1024 / 1024
    print(f"[源]   {src}  ({size:.1f} MB)")
    print(f"[目标] {args.target}")

    if args.dry_run:
        return

    Path(args.target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.target)
    print("[完成] 已复制。重启 webapp 即可让「自动识别区域」按钮生效。")


if __name__ == "__main__":
    main()
