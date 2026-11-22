@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================
echo  v10 网页演示：上传一张图，自动出框和区域
echo ============================================
echo 浏览器将自动打开 http://127.0.0.1:8000
echo 关闭本窗口即停止服务
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
:run_web
start "" http://127.0.0.1:8000
py -3 webapp.py --port 8000
goto finished
:run_local
start "" http://127.0.0.1:8000
"%PYTHON%" webapp.py --port 8000
:finished
pause
