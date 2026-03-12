from typing import Any, Dict

from mlflow import (
    active_run,
    end_run,
    log_metrics,
    log_params,
    set_experiment,
    set_tracking_uri,
    start_run,
)

from meta_learning.tracking.base import Tracker


class MLflowTracker(Tracker):
    def __init__(self, experiment_name: str, tracking_uri: str):
        set_tracking_uri(tracking_uri)
        set_experiment(experiment_name)

    def start_run(self, run_name: str):
        return start_run(run_name=run_name)

    def end_run(self) -> None:
        if active_run() is not None:
            end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        log_params(dict(params))

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        log_metrics(dict(metrics), step=step)

    def end_active_runs(self) -> None:
        while active_run() is not None:
            end_run()
