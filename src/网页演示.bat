@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v8 网页演示：上传一张图，自动出框和区域
echo ============================================
echo 浏览器将自动打开 http://127.0.0.1:8000
echo 关闭本窗口即停止服务
echo ============================================
start "" http://127.0.0.1:8000
"D:\yolo_env\Scripts\python.exe" webapp.py --model "D:\models\ebike26.pt" --port 8000
pause
