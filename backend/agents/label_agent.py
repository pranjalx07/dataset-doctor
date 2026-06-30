import pandas as pd
from typing import Dict, Any, List

class LabelAnalysisAgent:
    """
    An agent responsible for analyzing the target column (labels) of a dataset.
    
    It checks for missing labels, empty labels, counts unique labels, calculates
    duplicates, and evaluates the suitability of the dataset for classification tasks.
    """

    def analyze(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyzes the target column of a DataFrame to evaluate label quality and classification suitability.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.
            target_column (str): The name of the target column to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing label analysis results:
                - target_column (str): Name of the analyzed target column.
                - missing_labels (int): Count of missing (NaN/None) values in the column.
                - unique_labels (int): Count of unique label values (excluding missing).
                - empty_labels (int): Count of empty or whitespace-only string labels.
                - dataset_suitable (bool): True if the target column is suitable for classification, else False.
                - issues (List[str]): A list of descriptions for any identified label issues.

        Raises:
            ValueError: If df is None, is not a DataFrame, is empty, if target_column 
                        is invalid, or if target_column is not a string.
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

        issues: List[str] = []
        dataset_suitable = True

        # Calculate missing values
        missing_labels = int(df[target_column].isna().sum())
        total_samples = len(df)
        valid_samples = total_samples - missing_labels

        if missing_labels > 0:
            issues.append(f"Target column '{target_column}' contains {missing_labels} missing value(s).")
            if valid_samples == 0:
                issues.append(f"Target column '{target_column}' contains only missing values.")
                dataset_suitable = False

        # Calculate empty labels (empty strings or whitespace-only strings)
        empty_labels = 0
        if df[target_column].dtype == object or pd.api.types.is_string_dtype(df[target_column]):
            empty_labels = int(df[target_column].apply(lambda x: isinstance(x, str) and not x.strip()).sum())

        if empty_labels > 0:
            issues.append(f"Target column '{target_column}' contains {empty_labels} empty or whitespace-only label(s).")

        # Calculate unique labels
        unique_labels = int(df[target_column].dropna().nunique())

        if valid_samples > 0:
            if unique_labels < 2:
                issues.append(f"Target column '{target_column}' contains fewer than 2 unique classes ({unique_labels}). Classification requires at least 2 unique classes.")
                dataset_suitable = False

            # Calculate duplicate labels (repetition of labels across samples)
            duplicate_labels = int(df[target_column].dropna().duplicated().sum())
            if duplicate_labels == 0 and valid_samples > 1:
                issues.append(f"Target column '{target_column}' has no duplicate labels. Every sample has a unique label, which is not suitable for classification.")
                dataset_suitable = False

            # Check for high cardinality (regression target or identifier variable detection)
            # If there are a large number of unique labels relative to the sample size, it might be continuous or an ID
            if unique_labels > 100 or (unique_labels >= 10 and (unique_labels / valid_samples) > 0.5):
                # Float types with high cardinality are typically regression targets
                is_float = pd.api.types.is_float_dtype(df[target_column])
                target_type = "continuous/regression" if is_float else "high-cardinality/identifier"
                issues.append(f"Target column '{target_column}' has high cardinality ({unique_labels} unique labels for {valid_samples} samples), suggesting it may be a {target_type} target rather than a classification target.")
                dataset_suitable = False

        return {
            "target_column": target_column,
            "missing_labels": missing_labels,
            "unique_labels": unique_labels,
            "empty_labels": empty_labels,
            "dataset_suitable": dataset_suitable,
            "issues": issues
        }
