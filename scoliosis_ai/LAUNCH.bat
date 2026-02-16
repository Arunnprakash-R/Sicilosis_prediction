@echo off
cls
cd /d "%~dp0"

REM ========================================
REM UNIFIED SCOLIOSIS AI LAUNCHER
REM ONE FILE TO RUN EVERYTHING
REM ========================================

echo.
echo ========================================
echo   SCOLIOSIS AI - UNIFIED LAUNCHER
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Setting up Python environment...
    
    REM Try to find Python
    set PYTHON_CMD=python
    python --version >nul 2>&1
    
    if errorlevel 1 (
        echo [INFO] Creating virtual environment...
        if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe" (
            set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe
        ) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe" (
            set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe
        ) else if exist "C:\Python314\python.exe" (
            set PYTHON_CMD=C:\Python314\python.exe
        ) else (
            echo [ERROR] Python not found!
            echo Please install Python from: https://www.python.org/downloads/
            echo IMPORTANT: Check "Add Python to PATH" during installation
            pause
            exit /b 1
        )
    )
    
    "!PYTHON_CMD!" -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    
    echo [INFO] Installing dependencies...
    call venv\Scripts\pip.exe install --upgrade pip >nul 2>&1
    call venv\Scripts\pip.exe install -r requirements.txt >nul 2>&1
    
    if errorlevel 1 (
        echo [WARNING] Some packages failed to install
        echo Continuing anyway...
    )
    
    echo [OK] Setup complete!
    echo.
)

REM Launch the unified application
echo Launching Scoliosis AI...
echo.

venv\Scripts\python.exe launcher.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    echo.
    echo Try these solutions:
    echo   1. Delete 'venv' folder and run this file again
    echo   2. Reinstall Python from: https://www.python.org/downloads/
    echo   3. Check 'outputs/diagnosis/' for error logs
    echo.
    pause
    exit /b 1
)

exit /b 0
