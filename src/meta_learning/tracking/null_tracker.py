from contextlib import nullcontext
from typing import Any, Dict

from meta_learning.tracking.base import Tracker


class NullTracker(Tracker):
    def start_run(self, run_name: str):
        return nullcontext()

    def end_run(self) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        pass

    def end_active_runs(self) -> None:
        pass
