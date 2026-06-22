# System Architecture

## Overview
Dataset Doctor Agent is a multi-agent system analyzing uploaded datasets for ML readiness.

## Components
- **DataQualityAgent**: Handles quality analysis (null values, duplicates).
- **BiasAgent**: Checks class balance in a target column.
- **LabelAgent**: Analyzes label consistency.
- **ReportAgent**: Generates natural language summaries and recommendations.
- **Orchestrator**: Executes agents and consolidates results.
