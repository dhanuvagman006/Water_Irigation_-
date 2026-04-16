@echo off
echo =========================================
echo Starting Water Irrigation System
echo =========================================

echo [1/2] Activating Virtual Environment and Starting Backend Server...
start "Backend Server" cmd /k "call venv\Scripts\activate && cd backend && pip install -r requirements.txt && python -m uvicorn app.main:app --reload"

echo [2/2] Installing Dependencies and Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend && npm install && npm run dev"

echo.
echo Both servers are starting in separate windows!
echo It may take a moment to install dependencies first.
echo Close the newly opened command prompt windows to stop the servers.
