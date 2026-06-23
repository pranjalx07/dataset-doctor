import os
import tempfile
import pytest
import pandas as pd
from backend.utils.dataset_loader import load_csv, get_dataset_info

def test_load_csv_success():
    # Test loading a valid standard CSV
    df = load_csv("test_dataset.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "age" in df.columns

def test_load_csv_file_not_found():
    # Test loading a non-existent file path
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_csv("non_existent_file.csv")

def test_load_csv_empty_path():
    # Test empty path
    with pytest.raises(ValueError, match="path cannot be empty"):
        load_csv("")

def test_load_csv_directory_path():
    # Test passing a directory path
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="directory or not a valid file"):
            load_csv(tmpdir)

def test_load_csv_empty_file():
    # Test loading an empty file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
    try:
        with pytest.raises(ValueError, match="is empty"):
            load_csv(tmpfile_name)
    finally:
        if os.path.exists(tmpfile_name):
            os.remove(tmpfile_name)

def test_load_csv_alternative_encoding():
    # Test loading a file encoded in latin-1 (iso-8859-1) containing special characters
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
        
    try:
        # Write latin-1 data
        data = "name,val\ncrème,1\npiñata,2\n"
        with open(tmpfile_name, "w", encoding="latin-1") as f:
            f.write(data)
            
        # Ensure load_csv successfully uses fallback encoding
        df = load_csv(tmpfile_name)
        assert df.shape == (2, 2)
        assert df.iloc[0]["name"] == "crème"
        assert df.iloc[1]["name"] == "piñata"
    finally:
        if os.path.exists(tmpfile_name):
            os.remove(tmpfile_name)

def test_get_dataset_info_success():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "x"]
    })
    info = get_dataset_info(df)
    assert info["shape"] == {"rows": 3, "columns": 2}
    assert info["columns"] == ["a", "b"]
    assert "a" in info["summary_statistics"]
    assert "b" in info["categorical_info"]
