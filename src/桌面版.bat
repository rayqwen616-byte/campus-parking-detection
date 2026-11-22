@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo   校园违停检测 v9  ·  桌面版
echo ============================================
echo 正在启动本地服务，3 秒后自动打开窗口...
echo 关闭本黑窗口即停止服务。
echo ============================================
start "" /b cmd /c "timeout /t 3 >nul & start "" msedge --app=http://127.0.0.1:8000"
"D:\yolo_env\Scripts\python.exe" webapp.py --model "models\ebike26.pt" --port 8000
pause
