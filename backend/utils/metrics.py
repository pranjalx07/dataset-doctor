import pandas as pd
from typing import Dict, Any

def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validates that the input is a valid pandas DataFrame.
    
    Args:
        df: The object to validate.
        
    Raises:
        ValueError: If df is None or not a pandas DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

def calculate_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculates the count of missing values per column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Dict[str, int]: Mapping of column name to missing value count.
    """
    _validate_dataframe(df)
    if df.empty:
        return {str(col): 0 for col in df.columns}
    return {str(col): int(count) for col, count in df.isnull().sum().items()}

def calculate_missing_percentage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates the percentage of missing values per column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Dict[str, float]: Mapping of column name to missing value percentage.
    """
    _validate_dataframe(df)
    if df.empty:
        return {str(col): 0.0 for col in df.columns}
    total_rows = len(df)
    return {
        str(col): round(float((count / total_rows) * 100), 2)
        for col, count in df.isnull().sum().items()
    }

def calculate_duplicate_rows(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculates the count of duplicate rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Dict[str, int]: A dictionary containing the duplicate count.
    """
    _validate_dataframe(df)
    if df.empty:
        return {"count": 0}
    return {"count": int(df.duplicated().sum())}

def get_numerical_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Generates statistical summaries for numerical columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Dict[str, Dict[str, Any]]: Statistical summary for each numerical column.
    """
    _validate_dataframe(df)
    if df.empty:
        return {}
        
    numeric_df = df.select_dtypes(include=['number'])
    summary = {}
    
    for col in numeric_df.columns:
        col_series = numeric_df[col]
        desc = col_series.describe()
        
        stats_dict = {}
        for stat_name, val in desc.items():
            if pd.notnull(val):
                if stat_name == 'count':
                    stats_dict[stat_name] = int(val)
                else:
                    stats_dict[stat_name] = float(val)
            else:
                stats_dict[stat_name] = None
        
        if '50%' in stats_dict:
            stats_dict['median'] = stats_dict['50%']
            
        summary[str(col)] = stats_dict
        
    return summary

def get_categorical_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Generates unique counts and frequency previews for categorical, object, and boolean columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Dict[str, Dict[str, Any]]: Category summaries for each non-numerical column.
    """
    _validate_dataframe(df)
    if df.empty:
        return {}
        
    cat_df = df.select_dtypes(include=['object', 'category', 'bool'])
    summary = {}
    
    for col in cat_df.columns:
        col_series = cat_df[col]
        unique_count = int(col_series.nunique())
        
        value_counts = col_series.value_counts(dropna=False).head(5)
        top_values = {}
        for val, count in value_counts.items():
            val_str = str(val) if pd.notnull(val) else "Missing"
            top_values[val_str] = int(count)
            
        summary[str(col)] = {
            "unique_count": unique_count,
            "top_values": top_values
        }
        
    return summary
