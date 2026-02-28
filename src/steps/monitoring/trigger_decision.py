from zenml import step
import json
import subprocess
from src.utils.settings import SIMULATE_DRIFT

@step(enable_cache=False)
def trigger_decision(
    json_report_path: str,
    drift_threshold: float = 0.3,
    run_retrain: bool = True,
) -> bool:
    # Evidently json has drift summary; simplest heuristic:
    # count share of drifted columns from report json structure.
    with open(json_report_path, "r") as f:
        data = json.load(f)

    # try to find drift share (structure may vary by version)
    drift_share = 0.0
    try:
        # common location:
        drift_share = data["metrics"][0]["result"]["share_of_drifted_columns"]
    except Exception:
        drift_share = 0.0

    should_retrain = drift_share >= drift_threshold
    if SIMULATE_DRIFT:
        should_retrain = True
    print(f"[trigger_decision] drift_share={drift_share:.4f}, threshold={drift_threshold:.4f}, should_retrain={should_retrain}")

    if should_retrain and run_retrain:
        subprocess.run(["python", "-m", "src.pipelines.training_pipeline"], check=True)

    return should_retrain