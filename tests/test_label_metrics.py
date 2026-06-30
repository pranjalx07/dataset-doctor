import pytest
import pandas as pd
from backend.utils.label_metrics import (
    get_class_distribution,
    calculate_class_percentages,
    calculate_imbalance_ratio,
    count_missing_labels
)

def test_get_class_distribution():
    data = {"target": ["A", "B", "A", "A", "C", None]}
    df = pd.DataFrame(data)
    dist = get_class_distribution(df, "target")
    assert dist == {"A": 3, "B": 1, "C": 1}

def test_calculate_class_percentages():
    data = {"target": ["A", "B", "A", "A", "B", None]} # 5 non-null: 3 A (60%), 2 B (40%)
    df = pd.DataFrame(data)
    pcts = calculate_class_percentages(df, "target")
    assert pcts == {"A": 60.0, "B": 40.0}

def test_calculate_imbalance_ratio():
    data = {"target": ["A", "A", "A", "B", None]} # 3 A, 1 B -> ratio 3.0
    df = pd.DataFrame(data)
    ratio = calculate_imbalance_ratio(df, "target")
    assert ratio == 3.0

def test_count_missing_labels():
    data = {"target": ["A", None, "B", None]}
    df = pd.DataFrame(data)
    missing = count_missing_labels(df, "target")
    assert missing == 2

def test_invalid_inputs():
    df = pd.DataFrame({"target": [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Input DataFrame cannot be None"):
        get_class_distribution(None, "target")  # type: ignore

    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        get_class_distribution([1, 2, 3], "target")  # type: ignore

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        get_class_distribution(pd.DataFrame(), "target")

    with pytest.raises(ValueError, match="Target column name must be a string"):
        get_class_distribution(df, 123)  # type: ignore

    with pytest.raises(ValueError, match="Target column 'missing' not found in DataFrame"):
        get_class_distribution(df, "missing")
