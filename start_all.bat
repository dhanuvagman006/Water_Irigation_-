@echo off
title AquaAI Launcher

cd %~dp0

set /p TRAIN_MODELS="Do you want to train the models with the latest NASA dataset before starting? (y/N): "
if /i "%TRAIN_MODELS%"=="y" (
    echo [0/2] Training models... This may take several minutes.
    start /wait "AquaAI Model Training" cmd /c "cd backend && title AquaAI Training && python app\ml\train.py"
)

echo [1/2] Starting Backend Server...
start "AquaAI Backend" cmd /k "cd backend && pip install -r requirements.txt && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting Frontend...
cd frontend

REM Ensure Vite env vars exist (required for X-API-Key)
if not exist .env.local (
    if exist .env.example (
        echo Creating frontend\.env.local from .env.example...
        copy .env.example .env.local >nul
    ) else (
        echo Error: frontend\.env.example not found; cannot create .env.local
        exit /b 1
    )
)

call npm install
call npm run dev -- --open

pause