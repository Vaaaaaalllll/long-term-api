"""
S3 storage operations - download and upload parquet files.

Handles all S3 file operations. Based on storage_service.py from global-maritime-api.
Files are processed in memory (no local disk needed).
"""
import io
import pandas as pd
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def download_parquet_from_s3(s3_client, bucket, s3_key, local_path=None):
    """
    Download parquet file from S3 and load into DataFrame.
    
    Downloads directly to memory buffer, no local file needed.
    Can optionally save to local path if specified.
    """
    try:
        logger.info(f"Downloading from S3: {s3_key}")
        
        # Download to memory buffer
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, s3_key, buffer)
        buffer.seek(0)
        
        # Load parquet directly from buffer
        df = pd.read_parquet(buffer)
        logger.info(f"Downloaded {len(df)} rows from {s3_key}")
        
        # Optional: save to local file if path provided
        if local_path:
            df.to_parquet(local_path, index=False)
            logger.info(f"Saved to local path: {local_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3 Download Error: {str(e)}")


def upload_parquet_to_s3(s3_client, bucket, df, s3_key):
    """
    Upload DataFrame as parquet file to S3.
    
    Converts DataFrame to parquet in memory, then uploads.
    Results saved to bucket/username/filename.
    """
    try:
        logger.info(f"Uploading to S3: {s3_key}")
        
        # Convert DataFrame to parquet in memory
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # Upload from buffer
        s3_client.upload_fileobj(buffer, bucket, s3_key)
        
        logger.info(f"Upload completed: s3://{bucket}/{s3_key}")
        return {
            "status": "success",
            "file_location": f"s3://{bucket}/{s3_key}"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3 Upload Error: {str(e)}")

