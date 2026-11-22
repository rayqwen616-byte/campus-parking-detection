# -*- coding: utf-8 -*-
"""一键运行全部测试：python tests/run_all.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = [("judge_core 判定层自测(12项)", [str(ROOT / "judge_core.py")]),
        ("demo_v7 端到端冒烟(6项)", [str(ROOT / "tests" / "smoke_v7.py")]),
        ("voc2yolo 配对逻辑(6种情况)", [str(ROOT / "tests" / "test_voc2yolo.py")])]
bad = 0
for name, cmd in JOBS:
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    ok = p.returncode == 0
    bad += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(p.stdout[-1500:], p.stderr[-1500:])
sys.exit(1 if bad else 0)
