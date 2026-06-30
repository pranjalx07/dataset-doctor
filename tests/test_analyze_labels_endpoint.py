import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_analyze_labels_endpoint_success(tmp_path):
    # Create a temporary CSV file with multiple target classes
    csv_content = "name,label\nAlice,A\nBob,B\nCharlie,A\nDavid,B\nEva,A\n"
    temp_file = tmp_path / "test_labels.csv"
    temp_file.write_text(csv_content)
    
    with open(temp_file, "rb") as file:
        response = client.post(
            "/analyze-labels",
            files={"file": ("test_labels.csv", file, "text/csv")},
            data={"target_column": "label"}
        )
        
    assert response.status_code == 200
    report = response.json()
    
    # Assert return structure
    assert "bias_analysis" in report
    assert "label_analysis" in report
    
    bias = report["bias_analysis"]
    labels = report["label_analysis"]
    
    assert bias["target_column"] == "label"
    assert bias["dominant_class"] == "A"
    assert labels["unique_labels"] == 2
    assert labels["dataset_suitable"] is True

def test_analyze_labels_endpoint_invalid_column(tmp_path):
    csv_content = "name,label\nAlice,A\nBob,B\n"
    temp_file = tmp_path / "test_labels.csv"
    temp_file.write_text(csv_content)
    
    with open(temp_file, "rb") as file:
        response = client.post(
            "/analyze-labels",
            files={"file": ("test_labels.csv", file, "text/csv")},
            data={"target_column": "nonexistent"}
        )
        
    assert response.status_code == 400
    assert "not found in DataFrame" in response.json()["detail"]

def test_analyze_labels_endpoint_invalid_file_extension():
    response = client.post(
        "/analyze-labels",
        files={"file": ("test_labels.txt", b"some data", "text/plain")},
        data={"target_column": "label"}
    )
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]
