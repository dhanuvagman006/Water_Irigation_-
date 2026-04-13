import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.dependencies import get_db, get_model_loader
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.add = MagicMock()
    result_mock = MagicMock()
    scalars_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    scalars_mock.all.return_value = []
    session.execute.return_value = result_mock
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    yield session

@pytest.fixture
def mock_loader():
    loader = MagicMock()
    loader.models = {
        "rainfall/lstm": MagicMock(),
        "tank/lstm": MagicMock(),
        "irrigation/lstm": MagicMock()
    }
    loader.expected_models = ["lstm", "gru", "bilstm", "cnn_lstm", "transformer", "stacked_lstm"]
    # Provide dummy scalers or features so tests don't fail immediately
    yield loader

@pytest_asyncio.fixture
async def client(mock_db_session, mock_loader):
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_model_loader] = lambda: mock_loader
    headers = {"X-API-Key": "dev_super_secret_key_123"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=headers) as c:
        yield c
    app.dependency_overrides.clear()
