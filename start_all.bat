@echo off
echo =========================================
echo Starting Water Irrigation System
echo =========================================
echo Press Ctrl+C to stop both servers.
echo.


echo [0/3] Training Machine Learning Models (this may take a moment)...
call venv\Scripts\activate
python backend\app\ml\train.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed. Servers will not be started.
    pause
    exit /b %ERRORLEVEL%
)

echo [1/3] Starting Backend Server...
start /B "" cmd /c "call venv\Scripts\activate && cd backend && pip install -r requirements.txt && python -m uvicorn app.main:app --reload"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > NUL

echo [2/3] Installing Dependencies and Starting Frontend Server...
cd frontend
call npm install
npm run dev
