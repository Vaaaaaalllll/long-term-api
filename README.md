# Long-Term Trajectory Inference API

Simple API for long-term maritime trajectory prediction using diffusion models.

## Project Structure

```
long-term-api/
├── app/                    # Application layer
│   ├── main.py            # FastAPI app setup
│   ├── api/v1/            # API endpoints
│   │   └── routes.py      # Route handlers
│   ├── models/            # Request/Response schemas
│   │   └── schemas.py
│   └── core/              # Configuration
│       ├── config.py
│       └── dependencies.py  # S3 client initialization
├── services/               # Business logic
│   ├── inference_service.py
│   └── storage_service.py  # S3 operations
├── notebook/               # Example usage notebook
│   └── example_usage.ipynb
├── .env                   # Environment variables (create manually)
├── requirements.txt       # Dependencies
└── README.md
```

## Architecture

- **`app/`** - Handles HTTP requests/responses (API layer)
- **`services/`** - Contains business logic (domain layer)
- Clean separation: routes delegate to services

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# AWS S3 Configuration
S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
ENDPOINT_URL=your-endpoint-url

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Model Configuration
MODEL_CHECKPOINT_PATH=/path/to/unet.pt
MAP_IMAGE_PATH=/path/to/map.png
SEQ_LEN=144
BATCH_SIZE=64
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`

## API Endpoints

### POST `/api/v1/predict`

Download parquet file from S3, process it, and save results back to S3.

**Request Body:**
```json
{
  "s3_input_path": "username/raw_data/input_file.parquet",
  "username": "john_doe",
  "output_filename": "output_inference.parquet"
}
```

**Response:**
```json
{
  "status": "success",
  "input_file": "username/raw_data/input_file.parquet",
  "output_file": "john_doe/output_inference.parquet",
  "message": "Processed 1000 rows, saved to john_doe/output_inference.parquet"
}
```

**How it works:**
1. Downloads parquet file from `s3://bucket/{s3_input_path}`
2. Processes the data (runs inference)
3. Saves results to `s3://bucket/{username}/{output_filename}`

### GET `/api/v1/health`

Health check endpoint.

## Example Usage

See `notebook/example_usage.ipynb` for a complete example of how to use the API.

The notebook demonstrates:
- Health check
- Running predictions
- Batch processing multiple files
- Helper functions

## Next Steps

1. **Extract model classes** from your notebook into `services/`
2. **Implement model loading** in `InferenceService`
3. **Add inference logic** (heatmap generation, diffusion sampling)
4. **Test** with real trajectory data

## Development Notes

- **Input**: Parquet files fetched from S3
- **Output**: Results saved to S3 under `bucket/username/`
- **Configuration**: S3 credentials and model paths go in `.env`
- **ML Models**: Store in `models/` directory (gitignored) or `services/models/`
- **Structure Guide**: See `API_STRUCTURE_GUIDE.md` in the root directory for detailed explanation
- Keep it simple and maintainable
