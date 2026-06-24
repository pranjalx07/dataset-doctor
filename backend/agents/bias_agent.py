import pandas as pd
from typing import Dict, Any

class BiasDetectionAgent:
    """
    An agent responsible for detecting representational bias and class imbalance 
    in a pandas DataFrame target column.
    
    It analyzes the class distribution, calculates percentages, identifies the dominant class,
    determines the imbalance ratio (dominant class / minority class), and assesses the 
    imbalance risk level (Low, Medium, High).
    """

    def analyze(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyzes the target column of a DataFrame to detect class imbalance and bias.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.
            target_column (str): The name of the column to evaluate.

        Returns:
            Dict[str, Any]: A dictionary containing bias detection results:
                - target_column (str): Name of the analyzed column.
                - class_distribution (Dict[str, int]): Count of samples per class.
                - class_percentages (Dict[str, float]): Percentage distribution of each class.
                - dominant_class (str): The class with the largest count.
                - imbalance_ratio (float): Ratio of the dominant class to the minority class.
                - risk_level (str): Risk level based on the dominant class percentage:
                    - "Low" if dominant class < 60%
                    - "Medium" if dominant class between 60% and 80% (inclusive)
                    - "High" if dominant class > 80%

        Raises:
            ValueError: If df is None, is not a DataFrame, is empty, if target_column 
                        is invalid, or if the target column has no valid samples.
        """
        # Validate DataFrame
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Validate target_column
        if not isinstance(target_column, str):
            raise ValueError("Target column name must be a string.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        # Clean target column of null values
        clean_series = df[target_column].dropna()
        if clean_series.empty:
            raise ValueError(f"Target column '{target_column}' contains no valid (non-null) data.")

        # Calculate distributions
        class_counts = clean_series.value_counts()
        total_samples = int(clean_series.shape[0])

        class_distribution = {str(k): int(v) for k, v in class_counts.items()}
        class_percentages = {str(k): round(float((v / total_samples) * 100), 2) for k, v in class_counts.items()}
        
        # Identify dominant class
        dominant_class = str(class_counts.index[0])
        dominant_count = int(class_counts.iloc[0])
        minority_count = int(class_counts.iloc[-1])

        # Calculate imbalance ratio
        imbalance_ratio = round(float(dominant_count / minority_count), 2) if minority_count > 0 else 0.0

        # Assess risk level
        dominant_percentage = (dominant_count / total_samples) * 100
        if dominant_percentage < 60.0:
            risk_level = "Low"
        elif dominant_percentage <= 80.0:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "target_column": target_column,
            "class_distribution": class_distribution,
            "class_percentages": class_percentages,
            "dominant_class": dominant_class,
            "imbalance_ratio": imbalance_ratio,
            "risk_level": risk_level
        }
