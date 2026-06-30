import os
import json
import pandas as pd
from backend.utils.dataset_loader import load_csv
from backend.agents.bias_agent import BiasDetectionAgent
from backend.agents.label_agent import LabelAnalysisAgent

def run_label_and_bias_analysis() -> None:
    """
    Test script to load a sample CSV file, execute both BiasDetectionAgent
    and LabelAnalysisAgent on the target column, and print the formatted
    JSON results.
    """
    csv_path = "test_dataset.csv"
    target_column = "label"

    # 1. Load the CSV file
    print(f"Loading dataset from: {os.path.abspath(csv_path)}...")
    if not os.path.exists(csv_path):
        print(f"Error: Sample CSV file not found at '{csv_path}'")
        return
        
    try:
        df = load_csv(csv_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # 2. Run BiasDetectionAgent
    print(f"Running BiasDetectionAgent on target column '{target_column}'...")
    bias_agent = BiasDetectionAgent()
    try:
        bias_report = bias_agent.analyze(df, target_column)
    except Exception as e:
        print(f"Error executing BiasDetectionAgent: {e}")
        return

    # 3. Run LabelAnalysisAgent
    print(f"Running LabelAnalysisAgent on target column '{target_column}'...")
    label_agent = LabelAnalysisAgent()
    try:
        label_report = label_agent.analyze(df, target_column)
    except Exception as e:
        print(f"Error executing LabelAnalysisAgent: {e}")
        return

    # 4. Print results as formatted JSON
    combined_report = {
        "bias_detection": bias_report,
        "label_analysis": label_report
    }

    print("\n======================= ANALYSIS REPORT (JSON) =======================")
    print(json.dumps(combined_report, indent=2))
    print("======================================================================\n")

if __name__ == "__main__":
    run_label_and_bias_analysis()
