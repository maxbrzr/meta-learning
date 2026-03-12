from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Dict


class Tracker(ABC):
    @abstractmethod
    def start_run(self, run_name: str) -> AbstractContextManager[Any]:
        pass

    @abstractmethod
    def end_run(self) -> None:
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        pass

    @abstractmethod
    def end_active_runs(self) -> None:
        pass
