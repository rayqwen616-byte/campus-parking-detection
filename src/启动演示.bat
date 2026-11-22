@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v5 多帧投票 + 停留时长 + 冷却
echo ============================================
"D:\yolo_env\Scripts\python.exe" demo_v5.py --source "..\_samples\sample.mp4" --model "D:\models\ebike26.pt" --roi "0.0,0.45 0.75,0.35 0.95,0.75 0.85,1.0 0.0,1.0" --dwell 1 --vote 3 --cooldown 2 --out "_output"
echo.
echo 结果：_output\demo.mp4 与 events.csv
pause
