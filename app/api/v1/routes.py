"""
API route handlers.

This is the API layer - handles HTTP requests and delegates to services.
Keep routes thin, all business logic goes in services/.
"""
from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import InferenceRequest, InferenceResponse
from services.inference_service import InferenceService

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/predict", response_model=InferenceResponse)
async def predict(request: Request, data: InferenceRequest):
    """
    Main prediction endpoint.
    
    Takes S3 path to input parquet file, processes it, saves results back to S3.
    Input comes from request body (s3_input_path, username, output_filename).
    Results saved to bucket/username/output_filename.
    """
    # Get services from app state (initialized in main.py startup)
    service: InferenceService = request.app.state.inference_service
    s3_client = request.app.state.s3_client
    s3_bucket = request.app.state.s3_bucket
    
    try:
        # Delegate to service - all processing happens there
        result = service.process_from_s3(
            s3_client=s3_client,
            s3_bucket=s3_bucket,
            input_s3_key=data.s3_input_path,
            username=data.username,
            output_filename=data.output_filename
        )
        return InferenceResponse(**result)
    except ValueError as e:
        # Validation errors from service
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy"}

