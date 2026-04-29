@echo off
echo =========================================
echo Starting Water Irrigation System
echo =========================================
echo Press Ctrl+C to stop both servers.
echo.


echo [1/3] Activating Virtual Environment and Training Model...
call venv\Scripts\activate
python backend\app\ml\train.py

echo [2/3] Starting Backend Server...
start /B "" cmd /c "call venv\Scripts\activate && cd backend && pip install -r requirements.txt && python -m uvicorn app.main:app --reload"

echo [3/3] Installing Dependencies and Starting Frontend Server...
cd frontend
call npm install
npm run dev
