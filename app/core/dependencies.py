"""
Dependency initialization - creates S3 and Redis clients.

These are initialized once at startup and stored in app.state.
Same pattern as global-maritime-api repo.
"""
import boto3
import aioredis
import logging
from app.core.config import get_aws_config, get_redis_config

logger = logging.getLogger(__name__)


def initialize_s3_client():
    """
    Create and return S3 client using credentials from .env.
    
    Used for downloading input files and uploading results.
    """
    aws_config = get_aws_config()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_config['access_key_id'],
        aws_secret_access_key=aws_config['secret_access_key'],
        endpoint_url=aws_config['endpoint_url']
    )
    logger.info("S3 client initialized")
    return s3_client


def initialize_redis_client():
    """
    Create and return Redis client using config from .env.
    
    Currently initialized but not heavily used yet.
    Can be used for caching, session management, job tracking later.
    """
    redis_config = get_redis_config()
    redis = aioredis.from_url(
        f"redis://{redis_config['host']}:{redis_config['port']}",
        decode_responses=True,
        password=redis_config['password']
    )
    logger.info("Redis client initialized")
    return redis

