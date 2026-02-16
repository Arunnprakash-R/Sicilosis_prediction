@echo off
cls
cd /d "%~dp0"

REM ========================================
REM SCOLIOSIS AI - ENHANCED GUI LAUNCHER
REM ONE-CLICK START
REM ========================================

echo.
echo ========================================
echo   SCOLIOSIS AI - ENHANCED LAUNCHER
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [INFO] First time setup - Creating Python environment...
    echo.
    
    REM Try to find Python
    set PYTHON_CMD=python
    python --version >nul 2>&1
    
    if errorlevel 1 (
        echo [INFO] Looking for Python installation...
        if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe" (
            set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe
        ) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe" (
            set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe
        ) else if exist "C:\Python314\python.exe" (
            set PYTHON_CMD=C:\Python314\python.exe
        ) else (
            echo [ERROR] Python not found!
            echo.
            echo Please install Python from: https://www.python.org/downloads/
            echo IMPORTANT: Check "Add Python to PATH" during installation
            echo.
            pause
            exit /b 1
        )
    )
    
    echo [1/3] Creating virtual environment...
    "!PYTHON_CMD!" -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    
    echo [2/3] Upgrading pip...
    call venv\Scripts\pip.exe install --upgrade pip >nul 2>&1
    
    echo [3/3] Installing dependencies (this may take a few minutes)...
    call venv\Scripts\pip.exe install -r requirements.txt
    
    if errorlevel 1 (
        echo [WARNING] Some packages failed to install
        echo Continuing anyway...
    )
    
    echo.
    echo [OK] Setup complete!
    echo.
)

REM Launch the simplified GUI application
echo ========================================
echo   Launching Scoliosis AI Diagnosis
echo ========================================
echo.

venv\Scripts\python.exe launcher_simple.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Application failed to start
    echo ========================================
    echo.
    echo Try these solutions:
    echo   1. Delete 'venv' folder and run this file again
    echo   2. Reinstall Python: https://www.python.org/downloads/
    echo   3. Check error message above
    echo.
    pause
    exit /b 1
)

exit /b 0
