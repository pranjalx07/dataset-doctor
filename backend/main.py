import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.utils.dataset_loader import load_csv, get_dataset_info

app = FastAPI(title=settings.PROJECT_NAME)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload and output directories exist relative to the root/backend path
# Use absolute paths resolved against the directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
upload_path = os.path.join(BASE_DIR, settings.UPLOAD_DIR)
output_path = os.path.join(BASE_DIR, settings.OUTPUT_DIR)

os.makedirs(upload_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "ok", "project": settings.PROJECT_NAME}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only CSV files are supported."
        )

    # Resolve saving path
    file_path = os.path.join(upload_path, file.filename)
    
    # Save file to upload directory
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    finally:
        file.file.close()

    # Load file and extract info
    try:
        df = load_csv(file_path)
        dataset_info = get_dataset_info(df)
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV file: {str(e)}"
        )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "filepath": file_path,
        "dataset_info": dataset_info
    }
