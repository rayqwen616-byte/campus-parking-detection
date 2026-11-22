@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo 本版本没有推理程序，只有训练结果图表。
echo 即将打开本文件夹（6 张 png）
start "" "%~dp0"
pause
