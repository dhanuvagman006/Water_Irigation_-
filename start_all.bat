@echo off
echo =========================================
echo Starting Water Irrigation System
echo =========================================
echo Press Ctrl+C to stop both servers.
echo.

echo [1/2] Activating Virtual Environment and Starting Backend Server...
start /B "" cmd /c "call venv\Scripts\activate && cd backend && pip install -r requirements.txt && python -m uvicorn app.main:app --reload"

echo [2/2] Installing Dependencies and Starting Frontend Server...
cd frontend
call npm install
npm run dev
