@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v7 工程化：滞回 / gap 容错 / 骑手过滤
echo ============================================
"D:\yolo_env\Scripts\python.exe" demo_v7.py --source "..\_samples\sample.mp4" --model "D:\models\ebike26.pt" --zones "..\_samples\zones.json" --speed 10 --out "_output"
echo.
echo [跑测试]
"D:\yolo_env\Scripts\python.exe" tests\run_all.py
pause
