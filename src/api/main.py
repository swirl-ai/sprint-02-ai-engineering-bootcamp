from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
from httpx import AsyncClient

from contextlib import asynccontextmanager
from api.core.config import settings
from api.api.middleware import RequestIDMiddleware
from api.api.endpoints import api_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = AsyncClient(timeout=settings.DEFAULT_TIMEOUT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")

    yield

    logger.info("Application shutting down...")
    await client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "API"}