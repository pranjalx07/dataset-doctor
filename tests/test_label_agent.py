import pytest
import pandas as pd
from backend.agents.label_agent import LabelAnalysisAgent

def test_label_analysis_suitable_multiclass():
    # Arrange
    data = {
        "target": ["A", "B", "A", "C", "B", "C", "A", "B"] # 3 unique labels, duplicates exist
    }
    df = pd.DataFrame(data)
    agent = LabelAnalysisAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["target_column"] == "target"
    assert result["missing_labels"] == 0
    assert result["unique_labels"] == 3
    assert result["empty_labels"] == 0
    assert result["dataset_suitable"] is True
    assert len(result["issues"]) == 0

def test_label_analysis_with_missing_and_empty():
    # Arrange
    data = {
        "target": ["A", "B", " ", None, "A", "", "B", None] # 2 missing, 2 empty strings, A and B are valid
    }
    df = pd.DataFrame(data)
    agent = LabelAnalysisAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["target_column"] == "target"
    assert result["missing_labels"] == 2
    assert result["unique_labels"] == 4  # A, B, " ", "" are unique values before cleaning/checking empty?
    # Wait, unique_labels is nunique() excluding NaN. For pandas, " " and "" are unique, so including A and B, nunique is 4.
    # But wait, does empty string count as a unique label? Yes, nunique includes empty strings.
    assert result["empty_labels"] == 2
    # Since there are duplicates ("A" repeats, "B" repeats, empty repeats or there are 8 samples total),
    # let's check issues list
    assert any("missing value" in issue for issue in result["issues"])
    assert any("empty or whitespace-only" in issue for issue in result["issues"])

def test_label_analysis_unsuitable_single_class():
    # Arrange
    data = {
        "target": ["A", "A", "A", "A"]
    }
    df = pd.DataFrame(data)
    agent = LabelAnalysisAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dataset_suitable"] is False
    assert any("fewer than 2 unique classes" in issue for issue in result["issues"])

def test_label_analysis_unsuitable_all_unique():
    # Arrange
    data = {
        "target": ["A", "B", "C", "D"] # 4 samples, 4 unique, no duplicates
    }
    df = pd.DataFrame(data)
    agent = LabelAnalysisAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dataset_suitable"] is False
    assert any("no duplicate labels" in issue for issue in result["issues"])

def test_label_analysis_unsuitable_high_cardinality():
    # Arrange
    # Float column with distinct values (regression target)
    data = {
        "target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 1.1, 2.2] # 12 samples, 10 unique
    }
    df = pd.DataFrame(data)
    agent = LabelAnalysisAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dataset_suitable"] is False
    assert any("high cardinality" in issue for issue in result["issues"])

def test_label_analysis_unsuitable_all_missing():
    # Arrange
    data = {
        "target": [None, None, None]
    }
    df = pd.DataFrame(data)
    agent = LabelAnalysisAgent()

    # Act
    result = agent.analyze(df, "target")

    # Assert
    assert result["dataset_suitable"] is False
    assert any("contains only missing values" in issue for issue in result["issues"])

def test_label_analysis_invalid_inputs():
    agent = LabelAnalysisAgent()
    df = pd.DataFrame({"target": [1, 2, 1, 2]})

    # None DataFrame
    with pytest.raises(ValueError, match="Input DataFrame cannot be None."):
        agent.analyze(None, "target")  # type: ignore

    # Non-DataFrame
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame."):
        agent.analyze([1, 2, 3], "target")  # type: ignore

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame is empty."):
        agent.analyze(pd.DataFrame(), "target")

    # Non-string target_column
    with pytest.raises(ValueError, match="Target column name must be a string."):
        agent.analyze(df, 123)  # type: ignore

    # Missing target_column
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found in DataFrame."):
        agent.analyze(df, "nonexistent")
