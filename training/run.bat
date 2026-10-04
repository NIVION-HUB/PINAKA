@echo off
REM ============================================================
REM  PINAKA Master Launcher — Everything runs inside a venv
REM ============================================================
echo.
echo  ====================================================
echo   Project PINAKA — Aerospace Wake Word Training Pipeline
echo  ====================================================
echo.

cd /d "%~dp0"

REM --- Step 1: Create Virtual Environment ---
if not exist "venv" (
    echo [1/3] Creating isolated virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10+ first.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/3] Virtual environment already exists. Reusing.
)

REM --- Step 2: Install Dependencies Inside venv ---
echo [2/3] Installing dependencies inside venv (your PC stays clean)...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies. Check requirements.txt.
    pause
    exit /b 1
)
echo       All dependencies installed inside venv.

REM --- Step 3: Run the Master Training Pipeline ---
echo [3/3] Launching PINAKA Master Training Pipeline...
echo.
python master_train.py
if errorlevel 1 (
    echo.
    echo ERROR: Training pipeline failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo  ====================================================
echo   MISSION COMPLETE. Check the output/ folder.
echo  ====================================================
pause
