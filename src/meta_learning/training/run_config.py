from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class BaseRunConfig(ABC):
    @abstractmethod
    def create_experiment_id(self) -> str:
        raise NotImplementedError

    def to_tracking_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TinyHARRunConfig(BaseRunConfig):
    dataset_id: str
    epochs: int = 30
    patience: int = 5
    batch_size: int = 128
    learning_rate: float = 0.0003
    tracker_backend: Literal["mlflow", "wandb", "null"] = "mlflow"
    tracking_uri: str = "http://localhost:5001"

    def create_experiment_id(self) -> str:
        dataset = self.dataset_id.lower()
        return (
            f"{dataset}_tinyhar_ep{self.epochs}_pat{self.patience}"
            f"_bs{self.batch_size}_lr{self.learning_rate}"
        )


@dataclass
class MetaTrainRunConfig(BaseRunConfig):
    dataset_id: str
    epochs: int = 20
    patience: int = 5
    batch_size: int = 16
    learning_rate: float = 0.0003
    shots_per_class: int | tuple[int, int] = 10
    final_eval_episodes: int = 10
    set_encoder_variant: str = "mean"
    set_encoder_num_heads: int = 4
    class_aware: bool = False
    hypernetwork_variant: str = "task"
    num_train_batches_override: int | None = None
    tracker_backend: Literal["mlflow", "wandb", "null"] = "mlflow"
    tracking_uri: str = "http://localhost:5001"

    def __post_init__(self) -> None:
        if isinstance(self.shots_per_class, list):
            self.shots_per_class = tuple(self.shots_per_class)  # type: ignore

    def create_experiment_id(self) -> str:
        dataset = self.dataset_id.lower()
        return (
            f"{dataset}_lorametatinyhar_shots{self.shots_per_class}"
            f"_se{self.set_encoder_variant}_ca{int(self.class_aware)}"
            f"_hn{self.hypernetwork_variant}"
        )


@dataclass
class MetaPretrainedRunConfig(BaseRunConfig):
    dataset_id: str
    epochs: int = 30
    patience: int = 5
    batch_size: int = 12
    learning_rate: float = 0.0003
    shots_per_class: int | tuple[int, int] = (1, 16)
    final_eval_episodes: int = 64
    tinyhar_checkpoint_path: str | None = None
    set_encoder_variant: str = "mean"
    set_encoder_num_heads: int = 4
    class_aware: bool = False
    hypernetwork_variant: str = "task"
    num_train_batches_override: int | None = 128
    tracker_backend: Literal["mlflow", "wandb", "null"] = "mlflow"
    tracking_uri: str = "http://localhost:5001"

    def __post_init__(self) -> None:
        if isinstance(self.shots_per_class, list):
            self.shots_per_class = tuple(self.shots_per_class)  # type: ignore

    def create_experiment_id(self) -> str:
        dataset = self.dataset_id.lower()
        return (
            f"{dataset}_lorametatinyhar_pre_shots{self.shots_per_class}"
            f"_se{self.set_encoder_variant}_ca{int(self.class_aware)}"
            f"_hn{self.hypernetwork_variant}"
        )
