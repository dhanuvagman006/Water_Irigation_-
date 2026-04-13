from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class ModelMetricsBase(BaseModel):
    module: str
    model_name: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    nse: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None

class ModelMetricsCreate(ModelMetricsBase):
    pass

class ModelMetricsResponse(ModelMetricsBase):
    id: int
    evaluated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
