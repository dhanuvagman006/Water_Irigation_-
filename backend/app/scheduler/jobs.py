import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import date, timedelta
from app.database.base import AsyncSessionLocal
from app.services.nasa_service import nasa_service
from app.services.model_loader import model_loader
import numpy as np

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()


def _dummy_from_input_shape(input_shape):
    # Keras input_shape often includes None for batch dimension.
    # It may be a tuple or a list (multi-input models). We warm up the first input.
    shape = input_shape[0] if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)) else input_shape
    if not shape:
        return np.zeros((1, 1), dtype=np.float32)

    dims = []
    for dim in shape:
        dims.append(1 if dim is None else int(dim))
    return np.zeros(tuple(dims), dtype=np.float32)

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
    # Warmup models by passing dummy data matching each model's input shape.
    for module in model_loader.modules:
        for model_name in model_loader.expected_models:
            key = f"{module}/{model_name}"
            m = model_loader.models.get(key)
            if not m:
                continue
            try:
                dummy = _dummy_from_input_shape(getattr(m, "input_shape", None))
                m.predict(dummy, verbose=0)
            except Exception as e:
                logger.warning(f"Warmup failed for {key}: {e}")
                    
    logger.info("Weekly model warmup job completed")
