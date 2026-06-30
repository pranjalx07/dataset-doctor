import pandas as pd
from typing import Dict, Any, Optional
from backend.agents.data_quality_agent import DataQualityAgent
from backend.agents.bias_agent import BiasDetectionAgent
from backend.agents.label_agent import LabelAnalysisAgent

class OrchestratorAgent:
    """
    Consolidates dataset analysis by orchestrating three sub-agents:
    1. DataQualityAgent - analyzes overall data quality (missing values, duplicates, etc.).
    2. BiasDetectionAgent - analyzes class imbalance and bias in the target column.
    3. LabelAnalysisAgent - analyzes label consistency and suitability in the target column.
    """

    def __init__(self) -> None:
        self.quality_agent = DataQualityAgent()
        self.bias_agent = BiasDetectionAgent()
        self.label_agent = LabelAnalysisAgent()

    def analyze(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Executes all sub-agents on the input DataFrame and target column,
        handling exceptions gracefully and merging their outputs.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.
            target_column (str): The target column name to evaluate.

        Returns:
            Dict[str, Any]: A merged dictionary containing results from each agent:
                - "data_quality": Results from DataQualityAgent or error info.
                - "bias_detection": Results from BiasDetectionAgent or error info.
                - "label_analysis": Results from LabelAnalysisAgent or error info.
        """
        results: Dict[str, Any] = {
            "data_quality": {},
            "bias_detection": {},
            "label_analysis": {}
        }

        # 1. Run DataQualityAgent
        try:
            results["data_quality"] = self.quality_agent.analyze(df)
        except Exception as e:
            results["data_quality"] = {
                "error": f"DataQualityAgent failed: {str(e)}"
            }

        # 2. Run BiasDetectionAgent
        try:
            results["bias_detection"] = self.bias_agent.analyze(df, target_column)
        except Exception as e:
            results["bias_detection"] = {
                "error": f"BiasDetectionAgent failed: {str(e)}"
            }

        # 3. Run LabelAnalysisAgent
        try:
            results["label_analysis"] = self.label_agent.analyze(df, target_column)
        except Exception as e:
            results["label_analysis"] = {
                "error": f"LabelAnalysisAgent failed: {str(e)}"
            }

        return results
