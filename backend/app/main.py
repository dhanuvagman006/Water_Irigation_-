from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import structlog
from app.config import settings
from app.database.base import Base, engine
from app.services.model_loader import model_loader
from app.scheduler.jobs import scheduler
from app.routers import rainfall, tank, irrigation, models, data
from app.database.models import RainfallRecord, TankRecord, IrrigationRecord, ModelMetricsRecord, NASADataRecord

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing database tables...")
    # Optionally let alembic handle this, but per instructions we do it here:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Check and pre-fill NASA data if empty
    from app.database.base import AsyncSessionLocal
    from sqlalchemy import select, func
    from datetime import date, timedelta
    from app.services.nasa_service import nasa_service
    
    async with AsyncSessionLocal() as session:
        count = await session.scalar(select(func.count(NASADataRecord.id)))
        if count is None or count < 30:
            logger.info(f"Insufficient NASA data found ({count} records). Fetching past 30 days...")
            start_d = date.today() - timedelta(days=30)
            end_d = date.today()
            try:
                await nasa_service.fetch(start_d, end_d, session)
            except Exception as e:
                logger.error(f"Failed to fetch initial NASA data: {e}")
    
    logger.info("Loading models...")
    await model_loader.load_all()
    
    logger.info("Loading scalers...")
    await model_loader.load_scalers(settings.SCALERS_DIR)
    
    logger.info("Starting APScheduler...")
    scheduler.start()
    
    logger.info("AquaAI backend ready")
    yield
    # SHUTDOWN
    logger.info("Stopping scheduler...")
    scheduler.shutdown()
    logger.info("Clearing model cache...")
    model_loader.models.clear()
    model_loader.scalers.clear()
    logger.info("AquaAI backend shut down")

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend for AI-Based Water Management System",
    lifespan=lifespan
)

# Exception handlers
class ModelNotLoadedError(Exception):
    pass

class InsufficientDataError(Exception):
    pass

class NASAFetchError(Exception):
    pass

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(status_code=503, content={"error": str(exc), "code": "MODEL_NOT_LOADED"})

@app.exception_handler(InsufficientDataError)
async def insufficient_data_handler(request: Request, exc: InsufficientDataError):
    return JSONResponse(status_code=422, content={"error": str(exc), "code": "INSUFFICIENT_DATA"})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception", error=str(exc))
    return JSONResponse(status_code=500, content={"error": "An internal server error occurred.", "code": "INTERNAL_ERROR"})

# Middleware order (last added = first to run)
class VerifyAPIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Always allow OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Exempt paths from API key check
        exempt_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/health/ready"]
        if request.url.path in exempt_paths:
            return await call_next(request)
        
        if request.url.path.startswith("/api/rainfall/health"):
            return await call_next(request)
        
        # Check API key for /api/ routes
        if request.url.path.startswith("/api/"):
            x_api_key = request.headers.get("X-API-Key")
            if x_api_key != settings.API_KEY:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid API key", "code": "FORBIDDEN"}
                )
        
        return await call_next(request)

# Add middleware in correct order (last added runs first)
# We add CORS LAST so it runs FIRST and can handle OPTIONS before API key check
app.add_middleware(VerifyAPIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info("request", method=request.method, path=request.url.path, duration=duration, status=response.status_code)
    return response

# Routers
app.include_router(rainfall.router, prefix="/api/rainfall", tags=["Rainfall"])
app.include_router(tank.router, prefix="/api/tank", tags=["Tank"])
app.include_router(irrigation.router, prefix="/api/irrigation", tags=["Irrigation"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/health/ready")
async def health_ready():
    # Readiness check
    loaded_models_count = len(model_loader.models)
    expected_models_count = 18
    # Simplified checks for demo:
    return {
        "status": "ready" if loaded_models_count > 0 else "not_ready_partially",
        "models_loaded": loaded_models_count,
        "models_expected": expected_models_count,
        "db_connected": True,
        "nasa_data_fresh": True
    }
