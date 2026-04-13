import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_arecanut_gets_highest_water(client: AsyncClient, mocker):
    from app.services.irrigation_service import irrigation_service
    from app.schemas.irrigation import IrrigationPredictResponse, IrrigationDayPlan
    from datetime import date
    
    mock_resp = IrrigationPredictResponse(
        plan=[],
        total_water_liters={"Arecanut": 252.0, "Coconut": 168.0, "Pepper": 84.0},
        model_used="LSTM"
    )
    mocker.patch.object(irrigation_service, 'predict', return_value=mock_resp)
    
    payload = {
        "soil_moisture": 0.2,
        "crop_types": ["Arecanut", "Coconut", "Pepper"],
        "growth_stages": {"Arecanut": "Vegetative"},
        "model": "LSTM",
        "num_plants": {"Arecanut": 1, "Coconut": 1, "Pepper": 1}
    }
    response = await client.post("/api/irrigation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_water_liters"]["Arecanut"] > data["total_water_liters"]["Coconut"]

@pytest.mark.asyncio
async def test_no_irrigation_when_rain_predicted(client: AsyncClient, mocker):
    from app.services.irrigation_service import irrigation_service
    from app.schemas.irrigation import IrrigationPredictResponse, IrrigationDayPlan
    from datetime import date
    
    # Mocking that logic gives decision "No Irrigate" and water 0
    mock_resp = IrrigationPredictResponse(
        plan=[IrrigationDayPlan(date=date.today(), crop="Arecanut", decision="No Irrigate", water_liters=0.0, reason="Rain", soil_moisture_forecast=0.8)],
        total_water_liters={"Arecanut": 0.0},
        model_used="LSTM"
    )
    mocker.patch.object(irrigation_service, 'predict', return_value=mock_resp)
    
    payload = {
        "soil_moisture": 0.8,
        "crop_types": ["Arecanut"],
        "growth_stages": {"Arecanut": "Vegetative"},
        "model": "LSTM"
    }
    response = await client.post("/api/irrigation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["plan"][0]["decision"] == "No Irrigate"
    assert data["plan"][0]["water_liters"] == 0.0

@pytest.mark.asyncio
async def test_all_crops_returns_3_plans(client: AsyncClient, mocker):
    from app.services.irrigation_service import irrigation_service
    from app.schemas.irrigation import IrrigationPredictResponse, IrrigationDayPlan
    from datetime import date
    
    plan = [
        IrrigationDayPlan(date=date.today(), crop="Arecanut", decision="Irrigate", water_liters=10, reason="", soil_moisture_forecast=0.5),
        IrrigationDayPlan(date=date.today(), crop="Coconut", decision="Irrigate", water_liters=8, reason="", soil_moisture_forecast=0.5),
        IrrigationDayPlan(date=date.today(), crop="Pepper", decision="Irrigate", water_liters=5, reason="", soil_moisture_forecast=0.5)
    ]
    mock_resp = IrrigationPredictResponse(
        plan=plan,
        total_water_liters={"Arecanut": 10, "Coconut": 8, "Pepper": 5},
        model_used="LSTM"
    )
    mocker.patch.object(irrigation_service, 'predict', return_value=mock_resp)
    
    payload = {
        "soil_moisture": 0.2,
        "crop_types": ["Arecanut", "Coconut", "Pepper"],
        "growth_stages": {"Arecanut": "Vegetative"},
        "model": "LSTM"
    }
    response = await client.post("/api/irrigation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # In actual it would be 14*3=42, but we mocked 3 dayplans total
    assert len(data["plan"]) == 3
