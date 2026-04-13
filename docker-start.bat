REM Copy the .env.example file
copy .env.example backend\.env

REM Navigate to backend directory and start Docker Compose
cd backend
docker-compose up -d

REM Display URL information
echo.
echo ======================================
echo Water Irrigation System - Docker Started
echo ======================================
echo.
echo Frontend:  http://localhost:3000
echo Backend:   http://localhost:8000
echo API Docs:  http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop:      docker-compose down (from backend folder)
echo.
pause
