import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_predict_with_valid_inputs(client: AsyncClient, mocker):
    from app.services.tank_service import tank_service
    from app.schemas.tank import TankPredictResponse, TankDayPrediction
    from datetime import date
    
    mock_resp = TankPredictResponse(
        predictions=[TankDayPrediction(date=date.today(), level="Medium", percentage=50.0, estimated_liters=500.0)],
        alert=None,
        days_remaining=10,
        model_used="LSTM"
    )
    mocker.patch.object(tank_service, 'predict', return_value=mock_resp)
    
    payload = {
        "roof_area": 100.0,
        "tank_capacity": 1000.0,
        "current_level": 500.0,
        "daily_consumption": 50.0,
        "model": "LSTM",
        "rainfall_predictions": [0]*14
    }
    response = await client.post("/api/tank/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["days_remaining"] == 10
    assert data["model_used"] == "LSTM"

@pytest.mark.asyncio
async def test_low_tank_generates_alert(client: AsyncClient, mocker):
    from app.services.tank_service import tank_service
    from app.schemas.tank import TankPredictResponse, TankDayPrediction
    from datetime import date
    
    mock_resp = TankPredictResponse(
        predictions=[TankDayPrediction(date=date.today(), level="Low", percentage=10.0, estimated_liters=100.0)],
        alert="Tank LOW on Day 1",
        days_remaining=2,
        model_used="LSTM"
    )
    mocker.patch.object(tank_service, 'predict', return_value=mock_resp)
    
    payload = {
        "roof_area": 100.0,
        "tank_capacity": 1000.0,
        "current_level": 100.0,
        "daily_consumption": 50.0,
        "model": "LSTM",
        "rainfall_predictions": [0]*14
    }
    response = await client.post("/api/tank/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alert"] == "Tank LOW on Day 1"

@pytest.mark.asyncio
async def test_simulate_does_not_save_to_db(client: AsyncClient, mocker, mock_db_session):
    from app.services.tank_service import tank_service
    from app.schemas.tank import TankPredictResponse
    
    mock_resp = TankPredictResponse(predictions=[], alert=None, days_remaining=10, model_used="LSTM")
    mocker.patch.object(tank_service, 'predict', return_value=mock_resp)
    
    payload = {
        "roof_area": 100.0, "tank_capacity": 1000.0, "current_level": 500.0,
        "daily_consumption": 50.0, "model": "LSTM", "rainfall_predictions": [0]*14
    }
    response = await client.post("/api/tank/simulate", json=payload)
    assert response.status_code == 200
    # verify db.add was NOT called, unlike predict route
    mock_db_session.add.assert_not_called()
