import pandas as pd
import numpy as np
from typing import Dict, Any, List

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
        total_rows = len(df)
        missing_counts = df.isnull().sum()
        
        missing_info = {}
        for col in df.columns:
            count = int(missing_counts[col])
            percentage = float((count / total_rows) * 100) if total_rows > 0 else 0.0
            missing_info[str(col)] = {
                "count": count,
                "percentage": round(percentage, 2)
            }
        return missing_info

    def _get_duplicate_count(self, df: pd.DataFrame) -> int:
        """
        Calculates duplicate row counts.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            int: Number of duplicate rows.
        """
        return int(df.duplicated().sum())

    def _get_numerical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generates summary statistics for numerical columns.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, Dict[str, Any]]: Statistical summaries of numerical columns.
        """
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
            
            # Map '50%' to 'median' for clarity
            if '50%' in stats_dict:
                stats_dict['median'] = stats_dict['50%']
                
            summary[str(col)] = stats_dict
            
        return summary

    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generates key statistics for categorical, object, and boolean columns.
        
        Args:
            df (pd.DataFrame): The DataFrame.
            
        Returns:
            Dict[str, Dict[str, Any]]: Summary of categorical columns.
        """
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
