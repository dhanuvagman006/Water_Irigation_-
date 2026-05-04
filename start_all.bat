@echo off
setlocal enabledelayedexpansion

REM Get root directory (where this .bat is located)
set ROOT_DIR=%~dp0
set BACKEND_DIR=%ROOT_DIR%backend
set FRONTEND_DIR=%ROOT_DIR%frontend

echo =========================================
echo Starting Water Irrigation System
echo =========================================
echo Press Ctrl+C to stop everything
echo.

REM -------------------------------
REM [1/3] Install backend deps
REM -------------------------------
echo [1/3] Installing backend dependencies...

set PYTHON_CMD=py -3
if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    set PYTHON_CMD="%ROOT_DIR%.venv\Scripts\python.exe"
    echo Using virtual environment at .venv
)

%PYTHON_CMD% -m pip install -r "%BACKEND_DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo Backend dependency install failed.
    pause
    exit /b
)

REM -------------------------------
REM [2/3] Start backend server
REM -------------------------------
echo [2/3] Starting Backend Server...
start "backend" cmd /k "cd /d %BACKEND_DIR% && %PYTHON_CMD% -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001"

REM Small delay to avoid race condition
timeout /t 2 >nul

REM -------------------------------
REM [3/3] Install frontend deps
REM -------------------------------
echo [3/3] Installing frontend dependencies...
cd /d "%FRONTEND_DIR%"
call npm install
if %errorlevel% neq 0 (
    echo Frontend dependency install failed.
    pause
    exit /b
)

REM -------------------------------
REM Start frontend server
REM -------------------------------
echo [3/3] Starting Frontend Server...
start "frontend" cmd /k "cd /d %FRONTEND_DIR% && npm run dev -- --host 127.0.0.1"

echo.
echo =========================================
echo Backend:  http://127.0.0.1:8001
echo Frontend: http://127.0.0.1:5173
echo =========================================
echo.

REM Keep main window alive
pause