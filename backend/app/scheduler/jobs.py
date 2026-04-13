import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import date, timedelta
from app.database.base import AsyncSessionLocal
from app.services.nasa_service import nasa_service
from app.services.model_loader import model_loader
import numpy as np

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=0, minute=30)
async def daily_nasa_fetch():
    logger.info("Executing daily NASA fetch job")
    start = date.today() - timedelta(days=2)
    end = date.today() - timedelta(days=1)
    
    try:
        async with AsyncSessionLocal() as db:
            result = await nasa_service.fetch(start, end, db)
            logger.info(f"NASA Fetch Job Success: {result}")
    except Exception as e:
        logger.error(f"NASA Fetch Job Failed: {e}")

@scheduler.scheduled_job('cron', day_of_week='sun', hour=1, minute=0)
async def weekly_model_warmup():
    logger.info("Executing weekly model warmup job")
    # Warmup models by passing dummy data to keep them loaded computationally fast
    dummy_rain_features = np.zeros((1, 30, 8))
    dummy_tank_features = np.zeros((1, 5))
    dummy_irrigation_features = np.zeros((1, 7))
    
    for module, name_dict in [("rainfall", dummy_rain_features), ("tank", dummy_tank_features), ("irrigation", dummy_irrigation_features)]:
        for model_name in model_loader.expected_models:
            key = f"{module}/{model_name}"
            if key in model_loader.models:
                try:
                    m = model_loader.models[key]
                    m.predict(name_dict, verbose=0)
                except Exception as e:
                    logger.warning(f"Warmup failed for {key}: {e}")
                    
    logger.info("Weekly model warmup job completed")
