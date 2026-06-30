import pytest
import pandas as pd
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.report_agent import ReportAgent

def test_orchestrator_and_report_success():
    # Arrange
    # A standard classification dataset with duplicates, some missing values, and high class bias (High Risk)
    data = {
        "age": [25, 30, 35, 40, None, 25],
        "gender": ["M", "F", None, "M", "F", "M"],
        "label": [1, 1, 1, 1, 0, 1] # 5 ones, 1 zero
    }
    df = pd.DataFrame(data)
    
    # Act
    orchestrator = OrchestratorAgent()
    analysis_results = orchestrator.analyze(df, "label")
    
    # Assert Orchestrator structure
    assert "data_quality" in analysis_results
    assert "bias_detection" in analysis_results
    assert "label_analysis" in analysis_results
    
    quality = analysis_results["data_quality"]
    bias = analysis_results["bias_detection"]
    label = analysis_results["label_analysis"]
    
    assert quality["shape"] == {"rows": 6, "columns": 3}
    assert bias["risk_level"] == "High"
    assert label["dataset_suitable"] is True

    # Act - Report Generation
    report_agent = ReportAgent()
    report = report_agent.generate_report(analysis_results)
    
    # Assert Report Structure
    assert "executive_summary" in report
    assert "health_score" in report
    assert "key_findings" in report
    assert "recommended_preprocessing" in report
    assert "final_recommendation" in report
    
    # Ensure health score is within 0-100
    assert 0 <= report["health_score"] <= 100
    assert isinstance(report["health_score"], int)
    assert report["final_recommendation"] in ["Ready for ML", "Needs Cleaning", "Not Recommended"]

def test_orchestrator_handles_exceptions_gracefully():
    # Arrange: Pass invalid types/inputs to trigger internal agent exceptions
    orchestrator = OrchestratorAgent()
    
    # Act
    # Passing a list instead of a DataFrame will cause all agents to raise exceptions, which should be caught and returned as errors
    analysis_results = orchestrator.analyze([1, 2, 3], "label")  # type: ignore
    
    # Assert
    assert "error" in analysis_results["data_quality"]
    assert "error" in analysis_results["bias_detection"]
    assert "error" in analysis_results["label_analysis"]

def test_report_agent_error_handling():
    # Arrange
    analysis_results = {
        "data_quality": {"error": "Failed to analyze"},
        "bias_detection": {"error": "Failed to analyze"},
        "label_analysis": {"error": "Failed to analyze"}
    }
    
    # Act
    report_agent = ReportAgent()
    report = report_agent.generate_report(analysis_results)
    
    # Assert
    assert report["health_score"] < 50
    assert report["final_recommendation"] == "Not Recommended"
    assert any("error" in finding.lower() for finding in report["key_findings"])
