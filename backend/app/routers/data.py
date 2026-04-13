from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import date
from app.database.models import NASADataRecord
from app.dependencies import get_db
from app.services.nasa_service import nasa_service

router = APIRouter()

@router.get("/nasa/fetch")
async def fetch_nasa_data(start: date, end: date, db: AsyncSession = Depends(get_db)):
    try:
        result = await nasa_service.fetch(start, end, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nasa/latest")
async def get_latest_nasa(db: AsyncSession = Depends(get_db)):
    df = await nasa_service.get_recent(days=30, db=db)
    # Convert DF to dict format
    if df.empty:
        return []
    
    # We must format date to string if we return dicts, or use a Pydantic model
    records = df.to_dict(orient='records')
    for r in records:
        if 'date' in r and isinstance(r['date'], date):
            r['date'] = r['date'].isoformat()
    return records

@router.get("/status")
async def get_data_status(db: AsyncSession = Depends(get_db)):
    # 1. last_nasa_fetch (based on max fetched_at)
    stmt1 = select(func.max(NASADataRecord.fetched_at))
    result1 = await db.execute(stmt1)
    last_fetched = result1.scalar()
    
    # 2. records counts
    stmt2 = select(func.count(NASADataRecord.id))
    result2 = await db.execute(stmt2)
    count = result2.scalar()
    
    # 3. date coverage
    stmt3 = select(func.min(NASADataRecord.date), func.max(NASADataRecord.date))
    result3 = await db.execute(stmt3)
    min_d, max_d = result3.one_or_none() or (None, None)
    
    return {
        "last_nasa_fetch": last_fetched,
        "records_count": count,
        "date_coverage": {
            "start": min_d,
            "end": max_d
        }
    }
