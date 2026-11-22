@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v1 初版：基础检测（文件夹批量出图）
echo ============================================
"D:\yolo_env\Scripts\python.exe" demo.py --source "..\_samples" --model "D:\models\ebike26.pt" --keep ebike,bicycle --out "_output"
echo.
echo 结果在 _output 文件夹
pause
