from typing import Dict, Any, List

class ReportAgent:
    """
    An agent responsible for consolidating dataset analysis results
    and generating executive summaries, health scores, key findings,
    preprocessing recommendations, and final ML suitability assessments.
    """

    def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes consolidated agent analysis results to compile a comprehensive report.

        Args:
            analysis_results (Dict[str, Any]): Merged results from OrchestratorAgent containing
                                                data_quality, bias_detection, and label_analysis.

        Returns:
            Dict[str, Any]: A report dictionary with keys:
                - executive_summary (str): Brief overview of findings.
                - health_score (int): Health rating from 0 to 100.
                - key_findings (List[str]): Bulleted items outlining critical details.
                - recommended_preprocessing (List[str]): List of actionable data-prep steps.
                - final_recommendation (str): "Ready for ML", "Needs Cleaning", or "Not Recommended".
        """
        score = 100
        findings: List[str] = []
        preprocessing_steps: List[str] = []

        # 1. Evaluate Data Quality
        quality = analysis_results.get("data_quality", {})
        if "error" in quality:
            findings.append(f"Data Quality Analysis error: {quality['error']}")
            score -= 20
        else:
            # Missing values analysis
            missing_info = quality.get("missing_values", {})
            total_missing_pct = 0.0
            cols_with_missing = []
            for col, stats in missing_info.items():
                pct = stats.get("percentage", 0.0)
                if pct > 0:
                    cols_with_missing.append(col)
                    total_missing_pct += pct
            
            if cols_with_missing:
                avg_missing = total_missing_pct / len(missing_info)
                missing_penalty = min(25, int(avg_missing * 1.5))
                score -= missing_penalty
                findings.append(f"Missing values detected in {len(cols_with_missing)} column(s).")
                preprocessing_steps.append("Impute or remove missing values in features and targets.")

            # Duplicate rows analysis
            duplicates = quality.get("duplicate_count", 0)
            rows = quality.get("shape", {}).get("rows", 0)
            if duplicates > 0 and rows > 0:
                dup_pct = (duplicates / rows) * 100
                dup_penalty = min(15, int(dup_pct * 1.0))
                score -= dup_penalty
                findings.append(f"Duplicate rows detected: {duplicates} row(s) ({round(dup_pct, 2)}%).")
                preprocessing_steps.append("Deduplicate rows to prevent data leakage and overfitting.")

        # 2. Evaluate Bias Analysis
        bias = analysis_results.get("bias_detection", {})
        if "error" in bias:
            findings.append(f"Bias Detection Analysis error: {bias['error']}")
            score -= 15
        else:
            risk = bias.get("risk_level", "Low")
            dominant_class = bias.get("dominant_class")
            class_pcts = bias.get("class_percentages", {})
            dominant_pct = class_pcts.get(dominant_class, 0.0) if dominant_class else 0.0
            
            if risk == "Medium":
                score -= 15
                findings.append(f"Medium risk class imbalance detected: dominant class represents {dominant_pct}% of samples.")
                preprocessing_steps.append("Consider minor upsampling or downsampling strategies to balance classes.")
            elif risk == "High":
                score -= 30
                findings.append(f"High risk class imbalance detected: dominant class represents {dominant_pct}% of samples.")
                preprocessing_steps.append("Apply advanced balancing techniques (e.g. SMOTE, class weighting) to address severe imbalance.")

        # 3. Evaluate Label Analysis
        label = analysis_results.get("label_analysis", {})
        dataset_suitable = True
        if "error" in label:
            findings.append(f"Label Analysis error: {label['error']}")
            score -= 15
            dataset_suitable = False
        else:
            dataset_suitable = label.get("dataset_suitable", True)
            issues = label.get("issues", [])
            for issue in issues:
                findings.append(f"Label Issue: {issue}")
                score -= 10
            
        if not dataset_suitable:
            score = min(score, 40)
            findings.append("The target label is not suitable for classification training in its current state.")

        # Health score bounds enforcement
        score = max(0, min(100, score))

        # Determine final recommendation
        if score >= 80 and dataset_suitable:
            final_recommendation = "Ready for ML"
        elif score >= 50 and dataset_suitable:
            final_recommendation = "Needs Cleaning"
        else:
            final_recommendation = "Not Recommended"

        if not preprocessing_steps:
            preprocessing_steps.append("None. The dataset is already highly clean and ready.")

        # Generate Executive Summary
        if final_recommendation == "Ready for ML":
            summary = (
                f"The dataset is in excellent condition with a health score of {score}/100. "
                "Class representation is balanced, label quality is high, and data issues are minimal. "
                "The dataset is recommended for immediate machine learning model development."
            )
        elif final_recommendation == "Needs Cleaning":
            summary = (
                f"The dataset is partially suitable with a health score of {score}/100, "
                "but requires preprocessing. Issues such as missing values, duplicate records, or class "
                "imbalance should be resolved before training models."
            )
        else:
            summary = (
                f"The dataset has a critical health score of {score}/100 and is not suitable for ML "
                "in its current form. Critical problems such as label noise, cardinality mismatches, or "
                "severe representation bias must be addressed before proceeding."
            )

        return {
            "executive_summary": summary,
            "health_score": score,
            "key_findings": findings,
            "recommended_preprocessing": preprocessing_steps,
            "final_recommendation": final_recommendation
        }
