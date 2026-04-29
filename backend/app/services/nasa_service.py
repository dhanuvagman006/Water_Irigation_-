import httpx
import pandas as pd
from datetime import date, datetime, timedelta
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models import NASADataRecord
from app.config import settings

logger = logging.getLogger(__name__)

NASA_POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M,"
    "ALLSKY_SFC_SW_DWN,PS"
    "&community=RE"
    "&longitude={lon}&latitude={lat}"
    "&start={start}&end={end}"
    "&format=JSON"
)

class NASAService:
    async def fetch(self, start: date, end: date, db: AsyncSession) -> dict:
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        url = NASA_POWER_URL.format(
            lon=settings.NASA_LON,
            lat=settings.NASA_LAT,
            start=start_str,
            end=end_str
        )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
        timeseries = data.get("properties", {}).get("parameter", {})
        if not timeseries:
            logger.warning("No parameter data found in NASA API response")
            return {"records_fetched": 0}
            
        records_upserted = 0
        
        # Iterate over dates (NASA format YYYYMMDD)
        dates = list(timeseries.get("PRECTOTCORR", {}).keys())
        for d_str in dates:
            try:
                curr_date = datetime.strptime(d_str, "%Y%m%d").date()
                def get_val(param):
                    val = timeseries.get(param, {}).get(d_str, -999.0)
                    return None if val == -999.0 else val

                stmt = select(NASADataRecord).where(NASADataRecord.date == curr_date)
                result = await db.execute(stmt)
                record = result.scalars().first()
                
                if not record:
                    record = NASADataRecord(date=curr_date)
                    db.add(record)
                
                record.precipitation_mm = get_val("PRECTOTCORR")
                record.temp_max = get_val("T2M_MAX")
                record.temp_min = get_val("T2M_MIN")
                record.humidity = get_val("RH2M")
                record.wind_speed = get_val("WS2M")
                record.solar_radiation = get_val("ALLSKY_SFC_SW_DWN")
                record.pressure = get_val("PS")
                record.fetched_at = datetime.utcnow()
                
                records_upserted += 1
            except Exception as e:
                logger.error(f"Error processing NASA data for date {d_str}: {e}")
                
        await db.commit()
        logger.info(f"Fetched and upserted {records_upserted} records from NASA POWER API")
        return {"records_fetched": records_upserted, "start": start, "end": end}

    async def get_recent(self, days: int, db: AsyncSession) -> pd.DataFrame:
        # NOTE: NASA POWER daily data can lag by 1-2 days depending on timezone/update schedule.
        # For model inference we need the most recent `days` observations, not strictly the last
        # `days` calendar days ending today.
        stmt = select(NASADataRecord).order_by(NASADataRecord.date.desc()).limit(days)
        result = await db.execute(stmt)
        # Reverse back to ascending chronological order for feature engineering.
        records = list(result.scalars().all())[::-1]
        
        if not records:
            return pd.DataFrame()
            
        data = [{
            "date": r.date,
            "precipitation_mm": r.precipitation_mm,
            "temp_max": r.temp_max,
            "temp_min": r.temp_min,
            "humidity": r.humidity,
            "wind_speed": r.wind_speed,
            "solar_radiation": r.solar_radiation,
            "pressure": r.pressure
        } for r in records]
        
        df = pd.DataFrame(data)
        
        # Fill missing with forward-fill + seasonal median
        df.ffill(inplace=True)
        # Using simple median for entirely missing cols as fallback
        df.fillna(df.median(numeric_only=True), inplace=True)
        # For remaining NaNs
        df.fillna(0, inplace=True) 
        
        return df

nasa_service = NASAService()
