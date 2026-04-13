import datetime
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, func
from app.database.base import Base

class RainfallRecord(Base):
    __tablename__ = "rainfall_records"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True)
    actual_mm = Column(Float, nullable=True)
    predicted_mm = Column(Float, nullable=True)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class TankRecord(Base):
    __tablename__ = "tank_records"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    level_status = Column(String) # Low/Medium/Full
    percentage = Column(Float)
    roof_area = Column(Float)
    tank_capacity = Column(Float)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class IrrigationRecord(Base):
    __tablename__ = "irrigation_records"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    crop = Column(String) # Arecanut/Coconut/Pepper
    decision = Column(String) # Irrigate/No Irrigate/Monitor
    water_liters = Column(Float)
    soil_moisture = Column(Float)
    reason = Column(String)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ModelMetricsRecord(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, index=True)
    module = Column(String, index=True) # rainfall/tank/irrigation
    model_name = Column(String)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    nse = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.datetime.utcnow)

class NASADataRecord(Base):
    __tablename__ = "nasa_data"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True)
    precipitation_mm = Column(Float)
    temp_max = Column(Float)
    temp_min = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    solar_radiation = Column(Float)
    fetched_at = Column(DateTime, default=datetime.datetime.utcnow)
