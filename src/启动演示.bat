@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v6 多区域分级告警
echo ============================================
"D:\yolo_env\Scripts\python.exe" demo_v6.py --source "..\_samples\sample.mp4" --model "D:\models\ebike26.pt" --zones "zones.json" --vote 3 --speed 10 --out "_output"
echo.
echo 结果：_output\demo.mp4 与 events.csv（含等级列）
pause
