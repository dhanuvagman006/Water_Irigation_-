import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_predict_returns_14_days(client: AsyncClient, mocker):
    # Mock rainfall_service to avoid complex ML flow in unit testing the router
    from app.services.rainfall_service import rainfall_service
    from app.schemas.rainfall import RainfallPredictResponse, DayPrediction
    from datetime import date
    
    mock_resp = RainfallPredictResponse(
        predictions=[DayPrediction(date=date.today(), predicted_mm=10.0, confidence_low=8.0, confidence_high=12.0) for _ in range(14)],
        model_used="LSTM",
        generated_at=date.today()
    )
    
    mocker.patch.object(rainfall_service, 'predict', return_value=mock_resp)
    
    response = await client.post("/api/rainfall/predict", json={"model": "LSTM", "days": 14})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 14
    assert data["model_used"] == "LSTM"

@pytest.mark.asyncio
async def test_predict_invalid_model_returns_404(client: AsyncClient, mocker):
    from app.services.rainfall_service import rainfall_service
    from fastapi import HTTPException
    
    mocker.patch.object(rainfall_service, 'predict', side_effect=HTTPException(status_code=404, detail="Model missing/INVALID not loaded"))
    
    response = await client.post("/api/rainfall/predict", json={"model": "INVALID", "days": 14})
    assert response.status_code == 404
    assert "not loaded" in response.json()["detail"]

@pytest.mark.asyncio
async def test_metrics_returns_6_models(client: AsyncClient, mock_db_session, mocker):
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = [
        {"id": 1, "module": "rainfall", "evaluated_at": "2024-01-01T00:00:00", "model_name": "lstm"},
        {"id": 2, "module": "rainfall", "evaluated_at": "2024-01-01T00:00:00", "model_name": "gru"},
        {"id": 3, "module": "rainfall", "evaluated_at": "2024-01-01T00:00:00", "model_name": "bilstm"},
        {"id": 4, "module": "rainfall", "evaluated_at": "2024-01-01T00:00:00", "model_name": "cnn_lstm"},
        {"id": 5, "module": "rainfall", "evaluated_at": "2024-01-01T00:00:00", "model_name": "transformer"},
        {"id": 6, "module": "rainfall", "evaluated_at": "2024-01-01T00:00:00", "model_name": "stacked_lstm"}
    ]
    response = await client.get("/api/rainfall/metrics")
    assert response.status_code == 200
    assert len(response.json()) == 6

@pytest.mark.asyncio
async def test_history_returns_correct_range(client: AsyncClient, mock_db_session):
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = [
        {"date": "2024-01-01", "precipitation_mm": 5.0},
        {"date": "2024-01-02", "precipitation_mm": 10.0}
    ]
    response = await client.get("/api/rainfall/history?start=2024-01-01&end=2024-12-31")
    assert response.status_code == 200
    assert len(response.json()) == 2
