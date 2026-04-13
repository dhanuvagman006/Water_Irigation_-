# AquaAI Backend

This is the FastAPI backend for the AI-Based Water Management System.

## Architecture & Stack
- **Framework**: FastAPI + Uvicorn
- **Database**: SQLite (async via `aiosqlite` and `SQLAlchemy`, structured by `Alembic`)
- **ML Runtime**: TensorFlow + Joblib scalers
- **Scheduler**: APScheduler connected to NASA POWER REST API for automatic data pipeline updates

## Setup

1. **Install Dependencies**:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure Environment variables**:
Rename `.env.example` to `.env` and fill in necessary fields.

3. **Initialize Database**:
```bash
alembic upgrade head
```

4. **Launch Server**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
or via Docker:
```bash
docker-compose up --build
```

---

## Example cURL Requests

API expects requests to include an `X-API-Key` header mapped against the `.env` value.

### Health Check (No Auth)
```bash
curl -X GET http://localhost:8000/health
```

### Models Readiness
```bash
curl -X GET http://localhost:8000/api/models/list \
     -H "X-API-Key: dev_super_secret_key_123"
```

### 1) RAINFALL
```bash
curl -X POST http://localhost:8000/api/rainfall/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev_super_secret_key_123" \
     -d '{"model": "LSTM", "days": 14}'
```

### 2) TANK
```bash
curl -X POST http://localhost:8000/api/tank/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev_super_secret_key_123" \
     -d '{
       "roof_area": 100.0,
       "tank_capacity": 5000.0,
       "current_level": 3000.0,
       "daily_consumption": 200.0,
       "model": "LSTM"
     }'
```

### 3) IRRIGATION
```bash
curl -X POST http://localhost:8000/api/irrigation/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev_super_secret_key_123" \
     -d '{
       "soil_moisture": 0.45,
       "crop_types": ["Arecanut", "Coconut"],
       "growth_stages": {"Arecanut": "Vegetative", "Coconut": "Fruiting"},
       "num_plants": {"Arecanut": 50, "Coconut": 30},
       "model": "LSTM"
     }'
```

### Export Output
```bash
curl -X GET http://localhost:8000/api/irrigation/schedule/export \
     -H "X-API-Key: dev_super_secret_key_123" \
     --output schedule.csv
```
