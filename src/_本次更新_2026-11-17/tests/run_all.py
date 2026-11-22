# -*- coding: utf-8 -*-
"""一键运行全部测试：python tests/run_all.py"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

JOBS = [
    ("judge_core 判定层自测", [str(ROOT / "judge_core.py")]),
    ("demo_v8 端到端冒烟", [str(ROOT / "tests" / "smoke_v8.py")]),
    ("voc2yolo 配对逻辑", [str(ROOT / "tests" / "test_voc2yolo.py")]),
    ("webapp 后端单测", [str(ROOT / "tests" / "test_webapp.py")]),
    ("eval_zones 几何评测", [str(ROOT / "tests" / "test_eval_zones.py")]),
    ("predict_zones 掩码提取", [str(ROOT / "tests" / "test_predict_zones.py")]),
]

bad = 0
for name, cmd in JOBS:
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=ROOT, env=ENV)
    ok = p.returncode == 0
    bad += not ok
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if not ok:
        print((p.stdout or "")[-1500:])
        print((p.stderr or "")[-1500:])

print()
print("全部通过" if bad == 0 else "%d 项失败" % bad)
sys.exit(1 if bad else 0)
