import pytest
import pandas as pd
from backend.agents.bias_agent import BiasDetectionAgent

def test_bias_detection_low_risk():
    # Arrange
    data = {
        "target": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"] # A: 40%, B: 30%, C: 30%
    }
    df = pd.DataFrame(data)
    agent = BiasDetectionAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["target_column"] == "target"
    assert result["dominant_class"] == "A"
    assert result["class_distribution"] == {"A": 4, "B": 3, "C": 3}
    assert result["class_percentages"] == {"A": 40.0, "B": 30.0, "C": 30.0}
    assert result["imbalance_ratio"] == round(4 / 3, 2)
    assert result["risk_level"] == "Low"

def test_bias_detection_medium_risk():
    # Arrange
    data = {
        "target": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "C"] # A: 60%, B: 30%, C: 10%
    }
    df = pd.DataFrame(data)
    agent = BiasDetectionAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dominant_class"] == "A"
    assert result["class_percentages"]["A"] == 60.0
    assert result["risk_level"] == "Medium"
    assert result["imbalance_ratio"] == round(6 / 1, 2)

def test_bias_detection_high_risk():
    # Arrange
    # A: 90%, B: 10%
    data = {
        "target": ["A"] * 9 + ["B"]
    }
    df = pd.DataFrame(data)
    agent = BiasDetectionAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dominant_class"] == "A"
    assert result["class_percentages"]["A"] == 90.0
    assert result["risk_level"] == "High"
    assert result["imbalance_ratio"] == round(9 / 1, 2)

def test_bias_detection_binary():
    # Arrange
    data = {
        "target": [0, 1, 0, 1, 1] # 0: 40%, 1: 60%
    }
    df = pd.DataFrame(data)
    agent = BiasDetectionAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dominant_class"] == "1"
    assert result["class_distribution"] == {"1": 3, "0": 2}
    assert result["risk_level"] == "Medium"

def test_bias_detection_invalid_target_column():
    # Arrange
    df = pd.DataFrame({"target": [1, 2, 3]})
    agent = BiasDetectionAgent()

    # Act & Assert
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found in DataFrame."):
        agent.analyze(df, "nonexistent")

    with pytest.raises(ValueError, match="Target column name must be a string."):
        agent.analyze(df, 123)  # type: ignore

def test_bias_detection_empty_df_or_none():
    agent = BiasDetectionAgent()

    with pytest.raises(ValueError, match="Input DataFrame cannot be None."):
        agent.analyze(None, "target")  # type: ignore

    with pytest.raises(ValueError, match="Input must be a pandas DataFrame."):
        agent.analyze([1, 2, 3], "target")  # type: ignore

    with pytest.raises(ValueError, match="Input DataFrame is empty."):
        agent.analyze(pd.DataFrame(), "target")

def test_bias_detection_no_valid_data():
    df = pd.DataFrame({"target": [None, None]})
    agent = BiasDetectionAgent()

    with pytest.raises(ValueError, match="Target column 'target' contains no valid"):
        agent.analyze(df, "target")
