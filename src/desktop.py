# -*- coding: utf-8 -*-
"""
desktop.py —— 校园违停检测 · 桌面应用入口

原理：
    1. 后台启动本地服务（webapp.py，独立子进程，不弹黑窗口）
    2. 等服务就绪（轮询 /api/health）
    3. 用 pywebview 打开原生窗口（Windows 上是 WebView2）
    4. 关闭窗口时自动结束后台服务

用法：
    双击 校园违停检测.pyw        （无控制台窗口）
    python desktop.py           （有控制台，便于看日志）
    python desktop.py --test    （只启动服务并自检，不开窗口）
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller 打包后：以 exe 所在目录为应用根目录
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent
TITLE = "校园违停检测 v11"
WINDOW_W, WINDOW_H = 1360, 900

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def find_python():
    """优先用项目虚拟环境，其次当前解释器"""
    for c in (Path(r"D:\yolo_env\Scripts\python.exe"), BASE / ".venv" / "Scripts" / "python.exe"):
        if c.is_file():
            return str(c)
    return sys.executable


def pick_port(start=8000, tries=20):
    for p in range(start, start + tries):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("找不到可用端口")


def wait_ready(port, timeout=180):
    url = "http://127.0.0.1:%d/api/health" % port
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                data = json.loads(r.read().decode())
                if data.get("ok"):
                    return data
        except Exception:
            pass
        time.sleep(0.5)
    return None


def start_server(port, python, model=None, zone_model=None):
    cmd = [python, str(BASE / "webapp.py"), "--port", str(port)]
    if model:
        cmd += ["--model", model]
    if zone_model:
        cmd += ["--zone-model", zone_model]
    return subprocess.Popen(
        cmd, cwd=str(BASE), creationflags=CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--zone-model", default=None)
    ap.add_argument("--test", action="store_true", help="只启动服务并自检，不打开窗口")
    args = ap.parse_args()

    port = args.port or pick_port()
    python = find_python()
    print("[启动] 使用解释器：", python)
    print("[启动] 本地服务端口：", port)

    proc = start_server(port, python, args.model, args.zone_model)
    info = wait_ready(port)
    if info is None:
        proc.terminate()
        print("[失败] 服务启动超时，请检查 webapp.py 是否能单独运行")
        return 1

    print("[就绪] 模型：%s ｜ 区域模型：%s" % (info.get("model"), info.get("zone_model")))
    if args.test:
        proc.terminate()
        print("[自检通过] 服务可正常启动")
        return 0

    try:
        import webview
    except Exception as e:
        proc.terminate()
        print("[失败] 无法加载 pywebview：%s" % e)
        print("       可改用 桌面版.bat 或浏览器打开 http://127.0.0.1:%d" % port)
        return 2

    window = webview.create_window(
        TITLE, "http://127.0.0.1:%d/" % port,
        width=WINDOW_W, height=WINDOW_H, min_size=(1024, 700),
        text_select=True,
    )
    try:
        webview.start()          # 阻塞直到窗口关闭
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("[已退出] 后台服务已关闭")
    return 0


if __name__ == "__main__":
    sys.exit(main())
