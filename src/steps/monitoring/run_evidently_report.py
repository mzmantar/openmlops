from zenml import step
from zenml import get_step_context
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from typing import Tuple
from typing_extensions import Annotated
from src.utils.settings import MONITORING_DIR

@step(enable_cache=False)
def run_evidently_report(
    inference_path: str,
) -> Tuple[
    Annotated[str, "html_report_path"],
    Annotated[str, "json_report_path"],
]:
    df = pd.read_parquet(inference_path)

    if len(df) < 2:
        raise ValueError("Not enough rows in inference data to compute drift report.")

    # reference = first half, current = last half
    mid = len(df) // 2
    reference = df.iloc[:mid].copy()
    current = df.iloc[mid:].copy()

    # use only numeric columns for drift
    num_cols = [c for c in df.columns if c.startswith("proba_")]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[num_cols], current_data=current[num_cols])

    os.makedirs(MONITORING_DIR, exist_ok=True)
    html_path = os.path.join(MONITORING_DIR, "evidently_report.html")
    json_path = os.path.join(MONITORING_DIR, "evidently_report.json")

    report.save_html(html_path)
    report.save_json(json_path)

    context = get_step_context()
    context.add_output_metadata(
        output_name="html_report_path",
        metadata={
            "report_type": "evidently_html",
            "rows_reference": int(len(reference)),
            "rows_current": int(len(current)),
            "num_columns_monitored": int(len(num_cols)),
            "path": html_path,
        },
    )
    context.add_output_metadata(
        output_name="json_report_path",
        metadata={
            "report_type": "evidently_json",
            "rows_reference": int(len(reference)),
            "rows_current": int(len(current)),
            "num_columns_monitored": int(len(num_cols)),
            "path": json_path,
        },
    )

    return html_path, json_path