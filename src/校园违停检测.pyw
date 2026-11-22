# -*- coding: utf-8 -*-
"""双击启动桌面应用（无控制台窗口）"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)
try:
    from desktop import main
    sys.exit(main())
except Exception:
    import traceback
    log = Path(__file__).resolve().parent / '_desktop_error.log'
    log.write_text(traceback.format_exc(), encoding='utf-8')
    raise
