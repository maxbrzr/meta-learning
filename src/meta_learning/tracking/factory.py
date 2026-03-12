from typing import Literal

from meta_learning.tracking.base import Tracker
from meta_learning.tracking.mlflow_tracker import MLflowTracker
from meta_learning.tracking.null_tracker import NullTracker
from meta_learning.tracking.wandb_tracker import WandBTracker


def create_tracker(
    backend: Literal["mlflow", "wandb", "null"],
    experiment_name: str,
    tracking_uri: str | None = None,
) -> Tracker:
    match backend:
        case "mlflow":
            assert tracking_uri is not None, (
                "Tracking URI must be provided for MLflow backend"
            )
            return MLflowTracker(
                experiment_name=experiment_name, tracking_uri=tracking_uri
            )
        case "wandb":
            return WandBTracker(project=experiment_name)
        case "null":
            return NullTracker()
