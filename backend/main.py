import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.utils.dataset_loader import load_csv, get_dataset_info
from backend.agents.data_quality_agent import DataQualityAgent
from backend.agents.bias_agent import BiasDetectionAgent
from backend.agents.label_agent import LabelAnalysisAgent

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

@app.post("/analyze-quality")
async def analyze_quality(file: UploadFile = File(...)):
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

    # Load file and run DataQualityAgent
    try:
        df = load_csv(file_path)
        agent = DataQualityAgent()
        report = agent.analyze(df)
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to analyze CSV file: {str(e)}"
        )

    return report

@app.post("/analyze-labels")
async def analyze_labels(file: UploadFile = File(...), target_column: str = Form(...)):
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

    # Load file and run agents
    try:
        df = load_csv(file_path)
        bias_agent = BiasDetectionAgent()
        label_agent = LabelAnalysisAgent()
        
        bias_analysis = bias_agent.analyze(df, target_column)
        label_analysis = label_agent.analyze(df, target_column)
    except ValueError as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to analyze CSV file: {str(e)}"
        )
    finally:
        # Clean up uploaded file to avoid disk build-up
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

    return {
        "bias_analysis": bias_analysis,
        "label_analysis": label_analysis
    }
