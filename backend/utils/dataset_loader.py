import os
import pandas as pd
from typing import Dict, Any

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame with robust error handling.
    Checks file existence, tries multiple fallback encodings, handles parsing errors,
    and raises descriptive exceptions.
    
    Args:
        file_path (str): The absolute or relative path to the CSV file.
        
    Returns:
        pd.DataFrame: The parsed DataFrame.
        
    Raises:
        ValueError: If file path is empty, file path is a directory, empty file, or parsing/encoding errors occur.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path:
        raise ValueError("File path cannot be empty.")
        
    # Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
        
    # Ensure target path is a file, not a directory
    if not os.path.isfile(file_path):
        raise ValueError(f"The path '{file_path}' is a directory or not a valid file.")

    # Try common encodings to gracefully handle encoding errors
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'utf-16']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            # Try the next encoding in the list
            continue
        except pd.errors.EmptyDataError:
            raise ValueError(f"The CSV file at '{file_path}' is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file due to formatting issues: {str(e)}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while reading the CSV file: {str(e)}")
            
    raise ValueError(
        f"Unable to decode the CSV file at '{file_path}' using any of the standard encodings: "
        f"{', '.join(encodings_to_try)}."
    )

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates basic dataset metadata and stats.
    Ensures all types are native Python types for JSON serializability.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Dict[str, Any]: Basic metadata info.
    """
    shape = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }
    columns = list(df.columns)
    
    # Missing values analysis
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().mean() * 100).round(2).to_dict()
    
    missing_info = {
        col: {
            "count": int(missing_counts[col]),
            "percentage": float(missing_pct[col])
        }
        for col in columns
    }
    
    # Column datatypes
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Basic statistics for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    summary_stats = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()
        for col, stats in desc.items():
            summary_stats[col] = {k: float(v) if pd.notnull(v) else None for k, v in stats.items()}
            
    # Category preview for categorical/object columns
    categorical_df = df.select_dtypes(include=['object', 'category', 'bool'])
    categorical_info = {}
    for col in categorical_df.columns:
        val_counts = df[col].value_counts().head(5).to_dict()
        categorical_info[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in val_counts.items()}
        }

    return {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_info,
        "summary_statistics": summary_stats,
        "categorical_info": categorical_info
    }
