import json
import pytest
import pandas as pd
from backend.agents.data_quality_agent import DataQualityAgent

def test_data_quality_agent_success():
    # Arrange
    data = {
        "age": [25, 30, 35, 40, None],
        "gender": ["M", "F", None, "M", "F"],
        "label": [0, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    # Add a duplicate row to verify duplicate detection
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    agent = DataQualityAgent()
    
    # Act
    result = agent.analyze(df)
    
    # Assert
    assert result["shape"] == {"rows": 6, "columns": 3}
    assert set(result["columns"]) == {"age", "gender", "label"}
    assert "int" in result["dtypes"]["label"]
    
    # Missing values
    assert result["missing_values"]["age"]["count"] == 1
    # 1 out of 6 is ~16.67%
    assert result["missing_values"]["age"]["percentage"] == 16.67
    assert result["missing_values"]["gender"]["count"] == 1
    assert result["missing_values"]["label"]["count"] == 0
    
    # Duplicate count (the 1 row we duplicated)
    assert result["duplicate_count"] == 1
    
    # Numerical summaries (age and label are numeric)
    assert "age" in result["numerical_summary"]
    assert "label" in result["numerical_summary"]
    assert result["numerical_summary"]["age"]["count"] == 5  # 5 non-null age values (4 original + 1 duplicated)
    assert result["numerical_summary"]["age"]["mean"] == 31.0  # (25+30+35+40+25) / 5 = 31.0
    assert result["numerical_summary"]["age"]["min"] == 25.0
    assert result["numerical_summary"]["age"]["max"] == 40.0
    assert result["numerical_summary"]["age"]["median"] == 30.0  # Median of [25, 25, 30, 35, 40] is 30.0
    
    # Categorical summaries (gender is object/categorical)
    assert "gender" in result["categorical_summary"]
    assert result["categorical_summary"]["gender"]["unique_count"] == 2
    # Check value count inclusion
    top_vals = result["categorical_summary"]["gender"]["top_values"]
    assert top_vals["M"] == 3 # 2 originally, plus the 1 duplicated row [25, "M", 0]
    assert top_vals["F"] == 2
    assert top_vals["Missing"] == 1

    # Ensure JSON serializability
    try:
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
    except TypeError as e:
        pytest.fail(f"Result is not JSON serializable: {e}")

def test_data_quality_agent_empty_dataframe():
    # Arrange
    df = pd.DataFrame(columns=["col1", "col2"])
    agent = DataQualityAgent()
    
    # Act
    result = agent.analyze(df)
    
    # Assert
    assert result["shape"] == {"rows": 0, "columns": 2}
    assert result["columns"] == ["col1", "col2"]
    assert result["missing_values"]["col1"]["count"] == 0
    assert result["missing_values"]["col1"]["percentage"] == 0.0
    assert result["duplicate_count"] == 0
    assert result["numerical_summary"] == {}
    assert result["categorical_summary"] == {}
    
    # Ensure JSON serializability
    json.dumps(result)

def test_data_quality_agent_invalid_input():
    agent = DataQualityAgent()
    
    with pytest.raises(ValueError, match="Input DataFrame cannot be None"):
        agent.analyze(None)
        
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        agent.analyze([1, 2, 3])
