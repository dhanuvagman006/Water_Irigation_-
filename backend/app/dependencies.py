from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Request
from app.database.base import AsyncSessionLocal
from app.services.model_loader import model_loader, ModelLoader

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

async def get_model_loader(request: Request) -> ModelLoader:
    # We could attach it to app state in lifespan: request.app.state.model_loader
    # Setup for now returns the global instance
    return model_loader
