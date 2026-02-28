from zenml import pipeline
from src.steps.monitoring.load_latest_model import load_latest_model
from src.steps.monitoring.collect_inference_data import collect_inference_data
from src.steps.monitoring.run_evidently_report import run_evidently_report
from src.steps.monitoring.trigger_decision import trigger_decision
from src.steps.monitoring.store_monitoring_artifacts import store_monitoring_artifacts

@pipeline
def monitoring_pipeline():
    model = load_latest_model()
    inference_path = collect_inference_data(model)
    html_report_path, json_report_path = run_evidently_report(inference_path)
    _ = trigger_decision(json_report_path)
    _ = store_monitoring_artifacts(html_report_path, json_report_path)

if __name__ == "__main__":
    monitoring_pipeline()