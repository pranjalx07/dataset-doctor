import pandas as pd
from typing import Dict, Any, List
from backend.utils.metrics import (
    calculate_missing_values,
    calculate_missing_percentage,
    calculate_duplicate_rows,
    get_numerical_summary,
    get_categorical_summary
)

class DataQualityAgent:
    """
    An agent responsible for performing data quality analysis on a pandas DataFrame.
    
    It checks dataset dimensions, data types, missing values, duplicates, and generates
    summary statistics for numerical and categorical columns, returning a clean,
    JSON-serializable dictionary.
    """
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the full suite of data quality checks on the provided DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame to analyze.
            
        Returns:
            Dict[str, Any]: A JSON-serializable dictionary containing quality analysis.
            
        Raises:
            ValueError: If the input df is None or not a pandas DataFrame.
        """
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
            
        if df.empty:
            return self._analyze_empty_dataframe(df)
            
        return {
            "shape": self._get_shape(df),
            "columns": self._get_columns(df),
            "dtypes": self._get_dtypes(df),
            "missing_values": self._get_missing_values(df),
            "duplicate_count": self._get_duplicate_count(df),
            "numerical_summary": self._get_numerical_summary(df),
            "categorical_summary": self._get_categorical_summary(df)
        }

    def _analyze_empty_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Handles empty DataFrames gracefully, returning structured empty stats.
        
        Args:
            df (pd.DataFrame): The empty DataFrame.
            
        Returns:
            Dict[str, Any]: Dictionary representing stats for an empty DataFrame.
        """
        columns = [str(col) for col in df.columns]
        return {
            "shape": {
                "rows": 0,
                "columns": len(columns)
            },
            "columns": columns,
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {
                str(col): {
                    "count": 0,
                    "percentage": 0.0
                }
                for col in columns
            },
            "duplicate_count": 0,
            "numerical_summary": {},
            "categorical_summary": {}
        }

    def _get_shape(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Extracts the dimensions of the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, int]: Shape representation.
        """
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        }

    def _get_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Lists all column names.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            List[str]: List of column names.
        """
        return [str(col) for col in df.columns]

    def _get_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Maps each column to its string representation of data type.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, str]: Mapping of column to type.
        """
        return {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    def _get_missing_values(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculates count and percentage of missing values per column.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of column to count and percentage.
        """
        counts = calculate_missing_values(df)
        percentages = calculate_missing_percentage(df)
        return {
            str(col): {
                "count": counts[str(col)],
                "percentage": percentages[str(col)]
            }
            for col in df.columns
        }

    def _get_duplicate_count(self, df: pd.DataFrame) -> int:
        """
        Calculates duplicate row counts.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            int: Number of duplicate rows.
        """
        return calculate_duplicate_rows(df)["count"]

    def _get_numerical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generates summary statistics for numerical columns.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, Dict[str, Any]]: Statistical summaries of numerical columns.
        """
        return get_numerical_summary(df)

    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generates key statistics for categorical, object, and boolean columns.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, Dict[str, Any]]: Summary of categorical columns.
        """
        return get_categorical_summary(df)
