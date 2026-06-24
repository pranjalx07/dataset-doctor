import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_analyze_quality_endpoint_success(tmp_path):
    # Create a temporary CSV file
    csv_content = "name,age,gender\nJohn,28,M\nAlice,24,F\nBob,32,M\n,30,F\n"
    temp_file = tmp_path / "test_data.csv"
    temp_file.write_text(csv_content)
    
    with open(temp_file, "rb") as file:
        response = client.post(
            "/analyze-quality",
            files={"file": ("test_data.csv", file, "text/csv")}
        )
        
    assert response.status_code == 200
    report = response.json()
    
    # Assert shape
    assert report["shape"] == {"rows": 4, "columns": 3}
    # Assert columns
    assert report["columns"] == ["name", "age", "gender"]
    # Assert missing values: name has 1 missing value
    assert report["missing_values"]["name"]["count"] == 1
    
    # Clean up uploaded file from uploads dir if it exists
    upload_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "uploads", "test_data.csv")
    if os.path.exists(upload_file_path):
        os.remove(upload_file_path)

def test_analyze_quality_endpoint_invalid_file_extension():
    response = client.post(
        "/analyze-quality",
        files={"file": ("test_data.txt", b"some data", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]

def test_analyze_quality_endpoint_parsing_error(tmp_path):
    # Create an empty CSV file
    temp_file = tmp_path / "empty.csv"
    temp_file.write_text("")  # Empty file triggers EmptyDataError/ValueError in dataset_loader.load_csv
    
    with open(temp_file, "rb") as file:
        response = client.post(
            "/analyze-quality",
            files={"file": ("empty.csv", file, "text/csv")}
        )
        
    assert response.status_code == 422
    assert "Failed to analyze CSV file" in response.json()["detail"]
    
    upload_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "uploads", "empty.csv")
    if os.path.exists(upload_file_path):
        os.remove(upload_file_path)
