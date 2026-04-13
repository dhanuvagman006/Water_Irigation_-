from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from datetime import date
from io import StringIO
import csv
from app.schemas.irrigation import IrrigationPredictRequest, IrrigationPredictResponse
from app.schemas.metrics import ModelMetricsResponse
from app.database.models import IrrigationRecord, ModelMetricsRecord
from app.dependencies import get_db, get_model_loader
from app.services.irrigation_service import irrigation_service
from app.services.model_loader import ModelLoader

router = APIRouter()

@router.post("/predict", response_model=IrrigationPredictResponse)
async def predict(
    request: IrrigationPredictRequest,
    db: AsyncSession = Depends(get_db),
    loader: ModelLoader = Depends(get_model_loader)
):
    try:
        response = await irrigation_service.predict(request, loader, db)
        
        # Save results to DB
        for p in response.plan:
            record = IrrigationRecord(
                date=p.date,
                crop=p.crop,
                decision=p.decision,
                water_liters=p.water_liters,
                soil_moisture=p.soil_moisture_forecast,
                reason=p.reason,
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

@router.get("/predict/latest")
async def get_latest_predictions(crop: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    stmt = select(IrrigationRecord).order_by(IrrigationRecord.date.desc()).limit(14 * 3) # approx for 3 crops
    if crop:
        stmt = stmt.where(IrrigationRecord.crop == crop)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return records

@router.get("/metrics", response_model=List[ModelMetricsResponse])
async def get_metrics(db: AsyncSession = Depends(get_db)):
    stmt = select(ModelMetricsRecord).where(ModelMetricsRecord.module == "irrigation")
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/schedule/export")
async def get_schedule_export(db: AsyncSession = Depends(get_db)):
    stmt = select(IrrigationRecord).order_by(IrrigationRecord.date.desc()).limit(14 * 3)
    result = await db.execute(stmt)
    records = result.scalars().all()
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Crop", "Decision", "Water Liters", "Soil Moisture", "Reason", "Model Used"])
    
    for r in records:
        writer.writerow([r.date, r.crop, r.decision, r.water_liters, r.soil_moisture, r.reason, r.model_used])
        
    response = Response(content=output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=irrigation_schedule.csv"
    response.headers["Content-Type"] = "text/csv"
    return response
