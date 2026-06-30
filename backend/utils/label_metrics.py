import pandas as pd
from typing import Dict, Any, Union

def _validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    """
    Validates that the input is a valid pandas DataFrame and that the target column exists.
    
    Args:
        df: The object to validate.
        target_column: The target column name to validate.
        
    Raises:
        ValueError: If df is None, is not a pandas DataFrame, is empty, 
                    if target_column is not a string, or is not in the DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(target_column, str):
        raise ValueError("Target column name must be a string.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

def get_class_distribution(df: pd.DataFrame, target_column: str) -> Dict[str, int]:
    """
    Counts samples per class in the target column of a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        
    Returns:
        Dict[str, int]: A dictionary mapping class labels (as strings) to their sample count.
    """
    _validate_inputs(df, target_column)
    clean_series = df[target_column].dropna()
    if clean_series.empty:
        return {}
    
    class_counts = clean_series.value_counts()
    return {str(k): int(v) for k, v in class_counts.items()}

def calculate_class_percentages(df: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """
    Calculates the percentage distribution of each class in the target column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        
    Returns:
        Dict[str, float]: A dictionary mapping class labels (as strings) to their percentage (0-100),
                          rounded to two decimal places.
    """
    _validate_inputs(df, target_column)
    clean_series = df[target_column].dropna()
    total_samples = len(clean_series)
    if total_samples == 0:
        return {}
    
    class_counts = clean_series.value_counts()
    return {str(k): round(float((v / total_samples) * 100), 2) for k, v in class_counts.items()}

def calculate_imbalance_ratio(df: pd.DataFrame, target_column: str) -> float:
    """
    Calculates the imbalance ratio (dominant class count / minority class count) in the target column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        
    Returns:
        float: The ratio of the count of the majority class to the minority class.
               Returns 0.0 if there are no valid samples.
    """
    _validate_inputs(df, target_column)
    clean_series = df[target_column].dropna()
    if clean_series.empty:
        return 0.0
    
    class_counts = clean_series.value_counts()
    dominant_count = int(class_counts.iloc[0])
    minority_count = int(class_counts.iloc[-1])
    
    return round(float(dominant_count / minority_count), 2) if minority_count > 0 else 0.0

def count_missing_labels(df: pd.DataFrame, target_column: str) -> int:
    """
    Counts missing labels in the target column of a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        
    Returns:
        int: The number of missing (NaN/None) values in the target column.
    """
    _validate_inputs(df, target_column)
    return int(df[target_column].isna().sum())
