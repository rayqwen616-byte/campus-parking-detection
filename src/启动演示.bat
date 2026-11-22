@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v4 多边形禁停区 + 脚点判定
echo ============================================
"D:\yolo_env\Scripts\python.exe" demo_v4.py --source "..\_samples\sample_1.jpg" --model "D:\models\ebike26.pt" --roi "0.0,0.45 0.50,0.40 0.55,1.0 0.0,1.0" --out "_output"
echo.
echo 红区=禁停区，绿框=区域内目标，灰框=区域外
pause
