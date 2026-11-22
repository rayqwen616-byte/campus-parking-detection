@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v10 可信度升级版：视频判定演示
echo ============================================
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto run_local
set "PYTHON=D:\yolo_env\Scripts\python.exe"
if exist "%PYTHON%" goto run_local
where py >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python。请创建本地 .venv 或设置 D:\yolo_env。
  pause
  exit /b 1
)
py -3 demo_v8.py --source "samples\sample.mp4" --zones "samples\zones.json" --speed 10 --out "_output"
goto finished
:run_local
"%PYTHON%" demo_v8.py --source "samples\sample.mp4" --zones "samples\zones.json" --speed 10 --out "_output"
:finished
echo.
echo 如需运行测试，请手动执行 tests\run_all.py
echo 网页演示：双击「网页演示.bat」
pause
