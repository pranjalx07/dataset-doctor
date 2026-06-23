import pytest
import pandas as pd
from backend.utils.metrics import (
    calculate_missing_values,
    calculate_missing_percentage,
    calculate_duplicate_rows,
    get_numerical_summary,
    get_categorical_summary
)

@pytest.fixture
def sample_df():
    data = {
        "num1": [1.0, 2.0, None, 4.0],
        "num2": [10, 20, 30, 40],
        "cat1": ["a", "b", "a", None],
        "cat2": [True, False, True, True]
    }
    return pd.DataFrame(data)

def test_calculate_missing_values(sample_df):
    result = calculate_missing_values(sample_df)
    assert result == {
        "num1": 1,
        "num2": 0,
        "cat1": 1,
        "cat2": 0
    }

def test_calculate_missing_percentage(sample_df):
    result = calculate_missing_percentage(sample_df)
    assert result == {
        "num1": 25.0,
        "num2": 0.0,
        "cat1": 25.0,
        "cat2": 0.0
    }

def test_calculate_duplicate_rows(sample_df):
    # No duplicates
    assert calculate_duplicate_rows(sample_df) == {"count": 0}
    
    # Add duplicate
    df_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    assert calculate_duplicate_rows(df_dup) == {"count": 1}

def test_get_numerical_summary(sample_df):
    result = get_numerical_summary(sample_df)
    assert "num1" in result
    assert "num2" in result
    assert "cat1" not in result
    
    assert result["num1"]["count"] == 3
    assert result["num1"]["mean"] == 7.0 / 3.0
    assert result["num2"]["count"] == 4
    assert result["num2"]["mean"] == 25.0

def test_get_categorical_summary(sample_df):
    result = get_categorical_summary(sample_df)
    assert "cat1" in result
    assert "cat2" in result
    assert "num1" not in result
    
    assert result["cat1"]["unique_count"] == 2
    assert result["cat1"]["top_values"]["a"] == 2
    assert result["cat1"]["top_values"]["Missing"] == 1
    
    assert result["cat2"]["unique_count"] == 2

def test_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])
    assert calculate_missing_values(df) == {"a": 0, "b": 0}
    assert calculate_missing_percentage(df) == {"a": 0.0, "b": 0.0}
    assert calculate_duplicate_rows(df) == {"count": 0}
    assert get_numerical_summary(df) == {}
    assert get_categorical_summary(df) == {}

def test_invalid_input():
    for func in [
        calculate_missing_values,
        calculate_missing_percentage,
        calculate_duplicate_rows,
        get_numerical_summary,
        get_categorical_summary
    ]:
        with pytest.raises(ValueError, match="Input DataFrame cannot be None"):
            func(None)
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            func("not a dataframe")
