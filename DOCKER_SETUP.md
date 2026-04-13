# Water Irrigation System - Docker Setup

## Prerequisites
- Docker Desktop installed and running
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### Option 1: Using the batch script (Windows)
```batch
docker-start.bat
```

### Option 2: Manual Setup

1. **Copy environment file:**
   ```bash
   copy .env.example backend\.env
   ```

2. **Start Docker containers:**
   ```bash
   cd backend
   docker-compose up -d
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Services

The Docker Compose setup includes:
- **Backend**: FastAPI server on port 8000
- **Frontend**: React app on port 3000
- **Redis**: Cache database on port 6379

## Common Commands

### View logs
```bash
cd backend
docker-compose logs -f
```

### View specific service logs
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis
```

### Stop all services
```bash
docker-compose down
```

### Rebuild containers
```bash
docker-compose up -d --build
```

### Remove all data (containers, volumes)
```bash
docker-compose down -v
```

## Troubleshooting

### Port already in use
If ports 3000, 6379, or 8000 are already in use, either stop the conflicting services or modify the ports in `docker-compose.yml`.

### Containers not starting
Check logs: `docker-compose logs backend`

### Database issues
The SQLite database is persisted to `backend/aquaai.db`. To reset, delete this file and restart:
```bash
del backend\aquaai.db
docker-compose restart backend
```

### Frontend can't connect to backend
Ensure both services are in the same Docker network (they are by default). Check that the backend is fully started before the frontend tries to connect.

## Environment Variables

Edit `backend/.env` to modify:
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection URL
- `API_KEY`: API key for requests
- `SECRET_KEY`: Application secret key

## Production Deployment

For production, update the `.env` file with proper values and consider:
- Using a managed database instead of SQLite
- Setting up health checks
- Using environment-specific environment variables
- Proper CORS settings in backend
- HTTPS/SSL configuration
