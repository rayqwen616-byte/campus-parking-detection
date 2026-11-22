# -*- coding: utf-8 -*-
"""
eval_events.py —— 事件级评测：系统告警 vs 人工标注的真实违停事件

1) 先建一个标注文件 gt.csv（UTF-8，表头固定），每行是一次真实违停：
       区域,开始(s),结束(s)
       消防通道,35,410
   开始 = 车开始停放的时间，结束 = 车离开的时间；时间轴要和 events.csv 一致
   （评测请用 --speed 1 跑 demo_v7.py，时间就是视频真实秒）。
2) 用 demo_v7.py 跑同一段视频，得到 events.csv。
3) python eval_events.py --events runs/demo7/events.csv --gt gt.csv --duration 3600

输出 precision / recall / 平均告警延迟 / 每小时误报数。
告警时间落在 [开始 - tol, 结束 + tol] 且区域相同才算命中，一个标注事件只能被匹配一次。
"""
import argparse
import csv

from judge_core import evaluate_events


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=5.0, help="时间容差（秒）")
    ap.add_argument("--duration", type=float, default=None, help="评测视频总时长（秒），用于算每小时误报")
    args = ap.parse_args()

    pred = [{"zone": r["区域"], "time": float(r["时间(s)"])} for r in read(args.events)]
    gt = [{"zone": r["区域"], "start": float(r["开始(s)"]), "end": float(r["结束(s)"])} for r in read(args.gt)]
    m = evaluate_events(pred, gt, tol_s=args.tol, duration_s=args.duration)

    def f(x, fmt):
        return "n/a" if x is None else format(x, fmt)

    print(f"标注事件 {len(gt)} 个，系统告警 {len(pred)} 条")
    print(f"命中 TP={m['tp']}  误报 FP={m['fp']}  漏报 FN={m['fn']}")
    print(f"precision={f(m['precision'], '.3f')}  recall={f(m['recall'], '.3f')}")
    print(f"平均告警延迟={f(m['mean_latency_s'], '.1f')}s（相对车开始停放）")
    if "false_alarms_per_hour" in m:
        print(f"误报 {m['false_alarms_per_hour']:.2f} 次/小时")


if __name__ == "__main__":
    main()
