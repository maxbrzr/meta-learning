import os
from contextlib import AbstractContextManager
from typing import Any, Dict

import wandb
from dotenv import load_dotenv

from meta_learning.tracking.base import Tracker


class _WandbRunContext(AbstractContextManager[object]):
    def __init__(self, tracker: "WandBTracker", run_name: str):
        self.tracker = tracker
        self.run_name = run_name

    def __enter__(self) -> object:
        self.tracker._start_wandb_run(self.run_name)
        return self.tracker._run

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.tracker.end_run()


class WandBTracker(Tracker):
    def __init__(self, project: str):
        self.project = project
        self._run = None

        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

    def _start_wandb_run(self, run_name: str) -> None:
        self.end_run()
        self._run = wandb.init(project=self.project, name=run_name, reinit=True)

    def start_run(self, run_name: str):
        return _WandbRunContext(self, run_name)

    def end_run(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._run is None:
            return
        self._run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(dict(metrics), step=step)

    def end_active_runs(self) -> None:
        self.end_run()
