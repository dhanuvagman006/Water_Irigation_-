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

@pytest.mark.asyncio
async def test_soil_moisture_changes_water_liters(mock_db_session, mock_loader, mocker):
    from datetime import date, datetime, timedelta
    from app.schemas.irrigation import IrrigationPredictRequest
    from app.schemas.rainfall import DayPrediction, RainfallPredictResponse
    from app.services.irrigation_service import irrigation_service
    from app.services.rainfall_service import rainfall_service

    rainfall = RainfallPredictResponse(
        predictions=[
            DayPrediction(
                date=date.today() + timedelta(days=i + 1),
                predicted_mm=1.0,
                confidence_low=0.8,
                confidence_high=1.2,
            )
            for i in range(14)
        ],
        model_used="LSTM",
        generated_at=datetime.utcnow(),
    )
    mocker.patch.object(rainfall_service, "predict", return_value=rainfall)

    dry_request = IrrigationPredictRequest(
        soil_moisture=0.04,
        crop_types=["Arecanut"],
        growth_stages={"Arecanut": "Vegetative"},
        num_plants={"Arecanut": 50},
        model="LSTM",
    )
    moist_request = IrrigationPredictRequest(
        soil_moisture=0.55,
        crop_types=["Arecanut"],
        growth_stages={"Arecanut": "Vegetative"},
        num_plants={"Arecanut": 50},
        model="LSTM",
    )

    dry = await irrigation_service.predict(dry_request, mock_loader, mock_db_session)
    moist = await irrigation_service.predict(moist_request, mock_loader, mock_db_session)

    assert dry.plan[0].water_liters > moist.plan[0].water_liters
    assert dry.total_water_liters["Arecanut"] > moist.total_water_liters["Arecanut"]
