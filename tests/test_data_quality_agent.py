import os
import json
import pandas as pd
from backend.utils.dataset_loader import load_csv
from backend.agents.data_quality_agent import DataQualityAgent

def run_agent_test() -> None:
    """
    Test runner script that loads the workspace's test dataset, executes the 
    DataQualityAgent quality checks, and prints the formatted JSON analysis.
    """
    csv_path = "test_dataset.csv"
    
    # Verify file existence before running
    if not os.path.exists(csv_path):
        print(f"Error: Sample CSV file not found at '{csv_path}'")
        return
        
    # 1. Load the CSV file using our robust dataset loader
    print(f"Loading dataset from: {os.path.abspath(csv_path)}...")
    try:
        df = load_csv(csv_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
        
    # 2. Instantiate and execute the DataQualityAgent
    print("Initializing and running DataQualityAgent...")
    agent = DataQualityAgent()
    try:
        report = agent.analyze(df)
        print("Quality analysis completed.")
    except Exception as e:
        print(f"Error analyzing DataFrame: {e}")
        return
        
    # 3. Print the resulting JSON report
    print("\n======================= DATA QUALITY REPORT (JSON) =======================")
    print(json.dumps(report, indent=2))
    print("==========================================================================\n")

if __name__ == "__main__":
    run_agent_test()
