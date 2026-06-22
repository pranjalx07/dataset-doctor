import pandas as pd
from typing import Dict, Any

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates basic dataset metadata and stats.
    Ensures all types are native Python types for JSON serializability.
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
