@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v3 参数调优：对比两套参数
echo ============================================
echo [1/2] 默认参数 conf=0.35 iou=0.70
"D:\yolo_env\Scripts\python.exe" demo_v3.py --source "..\_samples\sample_1.jpg" --model "D:\models\ebike26.pt" --keep ebike,bicycle --conf 0.35 --iou 0.70 --out "_output\baseline"
echo.
echo [2/2] 优化参数 conf=0.25 iou=0.45
"D:\yolo_env\Scripts\python.exe" demo_v3.py --source "..\_samples\sample_1.jpg" --model "D:\models\ebike26.pt" --keep ebike,bicycle --conf 0.25 --iou 0.45 --out "_output\optimized"
echo.
echo 结果：_output\baseline 与 _output\optimized，可左右对比
pause
