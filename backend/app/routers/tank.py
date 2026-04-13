from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import date
from app.schemas.tank import TankPredictRequest, TankPredictResponse
from app.schemas.metrics import ModelMetricsResponse
from app.database.models import TankRecord, ModelMetricsRecord
from app.dependencies import get_db, get_model_loader
from app.services.tank_service import tank_service
from app.services.model_loader import ModelLoader

router = APIRouter()

@router.post("/predict", response_model=TankPredictResponse)
async def predict(
    request: TankPredictRequest,
    db: AsyncSession = Depends(get_db),
    loader: ModelLoader = Depends(get_model_loader)
):
    try:
        response = await tank_service.predict(request, loader, db)
        
        if response.predictions:
            first_day = response.predictions[0]
            record = TankRecord(
                date=first_day.date,
                level_status=first_day.level,
                percentage=first_day.percentage,
                roof_area=request.roof_area,
                tank_capacity=request.tank_capacity,
                model_used=response.model_used
            )
            db.add(record)
            await db.commit()
            
        return response
    except HTTPException as e:
        await db.rollback()
        raise e
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate", response_model=TankPredictResponse)
async def simulate(
    request: TankPredictRequest,
    db: AsyncSession = Depends(get_db),
    loader: ModelLoader = Depends(get_model_loader)
):
    # Same as predict but does not save to DB
    try:
        response = await tank_service.predict(request, loader, db)
        return response
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/latest")
async def get_latest_predictions(db: AsyncSession = Depends(get_db)):
    stmt = select(TankRecord).order_by(TankRecord.date.desc()).limit(14)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return records

@router.get("/metrics", response_model=List[ModelMetricsResponse])
async def get_metrics(db: AsyncSession = Depends(get_db)):
    stmt = select(ModelMetricsRecord).where(ModelMetricsRecord.module == "tank")
    result = await db.execute(stmt)
    return result.scalars().all()
