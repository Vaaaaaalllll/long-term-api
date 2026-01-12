"""
Main inference service - handles trajectory processing.

This is where the actual work happens. Business logic extracted from notebook.
Normalization params come from notebook Cell 13 (lat_mean, lon_mean, lat_std, lon_std).
"""
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
from services.storage_service import download_parquet_from_s3, upload_parquet_to_s3


class InferenceService:
    """
    Service for trajectory inference.
    
    This is the core business logic - processes trajectory data and runs inference.
    All the processing logic from the notebook goes here.
    """
    
    def __init__(self):
        # Normalization parameters from training data
        # These values come from notebook Cell 13 where we computed mean/std from traj_data
        # Using the exact values from the notebook: lat_mean=35.007248, lon_mean=135.92831, etc.
        self.lat_mean = 35.007248
        self.lon_mean = 135.92831
        self.lat_std = 1.5457377
        self.lon_std = 3.6006927
    
    def process_from_s3(self, s3_client, s3_bucket: str, input_s3_key: str, 
                      username: str, output_filename: str = None) -> Dict:
        """
        Main processing function - downloads from S3, processes, uploads results.
        
        This is the entry point called from routes. Handles the full pipeline:
        1. Download input parquet from S3
        2. Validate data
        3. Process (normalize, prepare for model, run inference)
        4. Upload results back to S3 under bucket/username/
        
        Args:
            s3_client: Boto3 S3 client
            s3_bucket: S3 bucket name
            input_s3_key: S3 path to input parquet file
            username: Username for output path (results go to bucket/username/)
            output_filename: Optional output filename (defaults to input_name_inference.parquet)
            
        Returns:
            Dictionary with processing results and S3 paths
        """
        # Step 1: Download input file from S3
        df_input = download_parquet_from_s3(s3_client, s3_bucket, input_s3_key)
        
        # Step 2: Validate data has required columns and valid ranges
        self._validate(df_input)
        
        # Step 3: Process data (normalize, prepare for model, run inference)
        df_output = self._process(df_input)
        
        # Step 4: Determine output path
        if output_filename is None:
            input_name = Path(input_s3_key).stem
            output_filename = f"{input_name}_inference.parquet"
        
        output_s3_key = f"{username}/{output_filename}"
        
        # Step 5: Upload results to S3
        upload_result = upload_parquet_to_s3(s3_client, s3_bucket, df_output, output_s3_key)
        
        return {
            "status": "success",
            "input_file": input_s3_key,
            "output_file": output_s3_key,
            "message": f"Processed {len(df_input)} rows, saved to {output_s3_key}"
        }
    
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process trajectory data and run inference.
        
        Currently returns input data as placeholder. When implementing actual inference:
        - Extract model classes from notebook Cell 19 (Guide_UNet, Model, etc.) to models/ directory
        - Load model checkpoint in model_service.py using MODEL_CHECKPOINT_PATH from .env
        - Generate heatmap (notebook Cell 17 - TrajectoryDataset._getitem__ method)
        - Load map image and fuse (notebook Cell 17 - fused_map calculation)
        - Prepare attributes Y (notebook Cell 8 - building Y array with trajectory metadata)
        - Run diffusion sampling (notebook Cell 23 - the p_xt loop with unet model)
        - Denormalize results (reverse normalization: pred * std + mean)
        """
        # Normalize coordinates (from notebook Cell 13)
        df_normalized = self._normalize(df.copy())
        
        # Prepare model input array (from notebook Cell 8 - building X array)
        # This creates the (seq_len, 8) feature array
        X = self._prepare_input(df_normalized)
        
        # Placeholder: return input data as "predictions"
        # Replace this section when implementing actual model inference
        df_result = df.copy()
        df_result["predicted_latitude"] = df["latitude"]
        df_result["predicted_longitude"] = df["longitude"]
        
        return df_result
    
    def _validate(self, df: pd.DataFrame):
        """
        Validate trajectory data has required columns and valid ranges.
        
        Based on validation from notebook Cell 6 (AIS sanity filters).
        """
        if df.empty:
            raise ValueError("Empty trajectory data")
        
        # Check required columns exist
        required_cols = ["latitude", "longitude"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Validate coordinate ranges (from notebook Cell 6)
        if not df["latitude"].between(-90, 90).all():
            raise ValueError("Latitude out of range [-90, 90]")
        if not df["longitude"].between(-180, 180).all():
            raise ValueError("Longitude out of range [-180, 180]")
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize latitude and longitude coordinates.
        
        Uses mean/std from training data (from notebook Cell 13).
        This is needed before feeding to the model.
        """
        df = df.copy()
        df["latitude"] = (df["latitude"] - self.lat_mean) / self.lat_std
        df["longitude"] = (df["longitude"] - self.lon_mean) / self.lon_std
        return df
    
    def _prepare_input(self, df: pd.DataFrame, seq_len: int = 144) -> np.ndarray:
        """
        Prepare input array for model - shape (seq_len, 8).
        
        Based on notebook Cell 8 where we build the X array.
        seq_len=144 means 12 hours at 5-minute intervals (from notebook Cell 5-6).
        
        Feature array contains:
        - latitude, longitude (normalized)
        - speed, course, heading, status (if available)
        - timestamp (epoch seconds)
        - offset (seconds from segment start)
        """
        # Pad or truncate to exactly seq_len points
        if len(df) < seq_len:
            # Pad with last value if too short
            last_row = df.iloc[-1:].copy()
            padding = pd.concat([last_row] * (seq_len - len(df)), ignore_index=True)
            df = pd.concat([df, padding], ignore_index=True)
        elif len(df) > seq_len:
            # Take first seq_len points if too long
            df = df.head(seq_len)
        
        # Build feature array (from notebook Cell 8 - building X)
        if "timestamp" in df.columns:
            # Convert timestamp to epoch seconds and compute offset
            timestamps = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            ts_raw = timestamps.astype("int64") // 10**9
            offset = (timestamps - timestamps.iloc[0]).dt.total_seconds()
            
            # Stack features: [lat, lon, speed, course, heading, status, timestamp, offset]
            # For now just use lat/lon if other columns missing
            if all(col in df.columns for col in ["speed", "course", "heading", "status"]):
                X = np.column_stack([
                    df[["latitude", "longitude", "speed", "course", "heading", "status"]].values,
                    ts_raw.values.astype(np.float32),
                    offset.values.astype(np.float32)
                ])
            else:
                # Fallback if optional columns missing
                X = np.column_stack([
                    df[["latitude", "longitude"]].values,
                    ts_raw.values.astype(np.float32),
                    offset.values.astype(np.float32)
                ])
        else:
            # No timestamp - just use lat/lon
            X = df[["latitude", "longitude"]].values
        
        return X.astype(np.float32)

