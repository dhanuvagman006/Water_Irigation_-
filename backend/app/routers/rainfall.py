from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import date
from app.schemas.rainfall import RainfallPredictRequest, RainfallPredictResponse
from app.schemas.metrics import ModelMetricsResponse
from app.database.models import RainfallRecord, ModelMetricsRecord, NASADataRecord
from app.dependencies import get_db, get_model_loader
from app.services.rainfall_service import rainfall_service
from app.services.model_loader import ModelLoader

router = APIRouter()

@router.post("/predict", response_model=RainfallPredictResponse)
async def predict(
    request: RainfallPredictRequest,
    db: AsyncSession = Depends(get_db),
    loader: ModelLoader = Depends(get_model_loader)
):
    try:
        response = await rainfall_service.predict(request, loader, db)
        
        # Save results to DB
        # We save the first prediction day just as a record, or could save all
        # For simplicity, saving the first day's prediction.
        if response.predictions:
            first_day = response.predictions[0]
            stmt = select(RainfallRecord).where(RainfallRecord.date == first_day.date)
            result = await db.execute(stmt)
            record = result.scalars().first()
            if record:
                record.predicted_mm = first_day.predicted_mm
                record.model_used = response.model_used
            else:
                record = RainfallRecord(
                    date=first_day.date,
                    predicted_mm=first_day.predicted_mm,
                    model_used=response.model_used
                )
                db.add(record)
            await db.commit()
            
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/latest")
async def get_latest_predictions(model: str = "LSTM", days: int = 14, db: AsyncSession = Depends(get_db)):
    # Returns last saved rainfall predictions from DB
    stmt = select(RainfallRecord).where(RainfallRecord.model_used == model).order_by(RainfallRecord.date.desc()).limit(days)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return records

@router.get("/metrics", response_model=List[ModelMetricsResponse])
async def get_metrics(db: AsyncSession = Depends(get_db)):
    stmt = select(ModelMetricsRecord).where(ModelMetricsRecord.module == "rainfall")
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/history")
async def get_history(start: date, end: date, db: AsyncSession = Depends(get_db)):
    stmt = select(NASADataRecord).where(NASADataRecord.date >= start, NASADataRecord.date <= end).order_by(NASADataRecord.date.asc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/health")
async def get_health(loader: ModelLoader = Depends(get_model_loader)):
    loaded_models = [k for k in loader.models.keys() if k.startswith("rainfall/")]
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "expected_models": loader.expected_models,
        "models": loaded_models
    }
