from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.schemas.metrics import ModelMetricsCreate, ModelMetricsResponse
from app.database.models import ModelMetricsRecord
from app.dependencies import get_db, get_model_loader
from app.services.model_loader import ModelLoader

router = APIRouter()

@router.get("/list")
async def list_models(loader: ModelLoader = Depends(get_model_loader)):
    # Group manually
    result = {"rainfall": [], "tank": [], "irrigation": []}
    for m in loader.expected_models:
        for mod in result.keys():
            key = f"{mod}/{m}"
            status = "loaded" if key in loader.models else "failed"
            result[mod].append({"name": m, "status": status})
    return result

@router.get("/metrics/{module}", response_model=List[ModelMetricsResponse])
async def get_module_metrics(module: str, db: AsyncSession = Depends(get_db)):
    if module not in ["rainfall", "tank", "irrigation"]:
        raise HTTPException(status_code=400, detail="Invalid module")
    # Order by ID to ensure latest is last (for frontend Map logic)
    stmt = select(ModelMetricsRecord).where(ModelMetricsRecord.module == module).order_by(ModelMetricsRecord.id.asc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/metrics/save", response_model=ModelMetricsResponse)
async def save_metrics(metrics: ModelMetricsCreate, db: AsyncSession = Depends(get_db)):
    record = ModelMetricsRecord(**metrics.model_dump())
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record

@router.get("/compare")
async def compare_models(db: AsyncSession = Depends(get_db)):
    stmt = select(ModelMetricsRecord)
    result = await db.execute(stmt)
    records = result.scalars().all()
    
    comp = {}
    for r in records:
        if r.module not in comp:
            comp[r.module] = []
        comp[r.module].append({
            "model_name": r.model_name,
            "rmse": r.rmse,
            "mae": r.mae,
            "r2": r.r2,
            "nse": r.nse,
            "accuracy": r.accuracy,
            "f1": r.f1
        })
    return comp
