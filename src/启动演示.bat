@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v8 合并版：判定演示 + 全部测试
echo ============================================
"D:\yolo_env\Scripts\python.exe" demo_v8.py --source "..\_samples\sample.mp4" --model "D:\models\ebike26.pt" --zones "..\_samples\zones.json" --speed 10 --out "_output"
echo.
echo [跑测试]
"D:\yolo_env\Scripts\python.exe" tests\run_all.py
echo.
echo 想用网页演示，双击「网页演示.bat」
pause
