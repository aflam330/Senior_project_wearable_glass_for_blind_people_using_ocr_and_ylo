@echo off
REM ============================================================================
REM  Smart Glass — Windows Test Launcher
REM
REM  Laptop/webcam dev runner for test_windows.py (gTTS + pygame, tkinter UI,
REM  keyboard controls instead of GPIO). First run creates a local venv and
REM  installs dependencies; later runs just activate it and launch the app.
REM
REM  This file is Windows-only tooling — it does not touch main.py, config.py,
REM  install.sh, smart_glass.service, or any other Raspberry Pi deployment file.
REM
REM  Usage: double-click this file, or run it from a terminal.
REM ============================================================================
setlocal

REM UTF-8 console code page so Bangla text prints correctly in logs
chcp 65001 >nul

cd /d "%~dp0"

set "VENV_DIR=.venv"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ from python.org and try again.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment in "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Using existing virtual environment "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

echo [2/3] Installing/updating dependencies (this can take a while on first run) ...
python -m pip install --upgrade pip wheel setuptools >nul
pip install gtts pygame opencv-python easyocr ultralytics torch torchvision numpy Pillow
if errorlevel 1 (
    echo [ERROR] Dependency installation failed — check the messages above.
    pause
    exit /b 1
)

echo.
echo [3/3] Launching Smart Glass Windows Test ...
echo        Controls: M = switch mode   A = read/detect   + / - = volume   Q = quit
echo.
python test_windows.py

echo.
echo Test session ended.
pause
