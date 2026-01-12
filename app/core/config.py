"""
Configuration management - loads settings from .env file.

All config values come from environment variables.
Bucket names, credentials, model paths go here - not in code.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
BASE_DIR = Path(__file__).parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def get_s3_bucket():
    """Get S3 bucket name from .env."""
    return os.getenv("S3_BUCKET")


def get_aws_config():
    """
    Get AWS S3 configuration from .env.
    
    Used to initialize S3 client in dependencies.py.
    """
    return {
        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'endpoint_url': os.getenv('ENDPOINT_URL')
    }


def get_redis_config():
    """
    Get Redis configuration from .env.
    
    Used to initialize Redis client in dependencies.py.
    """
    return {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'password': os.getenv('REDIS_PASSWORD')
    }


def get_model_config():
    """
    Get model configuration from .env.
    
    Model paths and processing parameters.
    These will be used when we implement actual model loading.
    """
    return {
        "checkpoint_path": os.getenv("MODEL_CHECKPOINT_PATH"),
        "map_image_path": os.getenv("MAP_IMAGE_PATH"),
        "seq_len": int(os.getenv("SEQ_LEN", 144)),  # 144 = 12 hours at 5-min intervals
        "batch_size": int(os.getenv("BATCH_SIZE", 64)),
    }

