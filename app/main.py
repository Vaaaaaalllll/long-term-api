"""
Main FastAPI application entry point.

This sets up the API server and initializes all dependencies (S3, Redis, services).
Structure follows the pattern from global-maritime-api repo.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import routes
from app.core.config import get_s3_bucket
from app.core.dependencies import initialize_s3_client, initialize_redis_client
from services.inference_service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Long-Term Trajectory Inference API",
    version="1.0.0"
)

# CORS middleware - allows frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """
    Initialize all services when API starts.
    
    Sets up S3 client, Redis connection, and inference service.
    These are stored in app.state so routes can access them.
    """
    app.state.s3_client = initialize_s3_client()
    app.state.s3_bucket = get_s3_bucket()
    app.state.redis = initialize_redis_client()
    await app.state.redis.ping()
    app.state.inference_service = InferenceService()
    logger.info(f"API started - S3 bucket: {app.state.s3_bucket}, Redis connected")


@app.get("/")
async def root():
    return {"message": "Long-Term Trajectory Inference API", "docs": "/docs"}


app.include_router(routes.router)

