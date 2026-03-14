import json
import os
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from whar_datasets import (
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

import hydra
from meta_learning.models.tiny_har import TinyHAR
from meta_learning.tracking import create_tracker
from meta_learning.training.run_config import TrainRunConfig
from meta_learning.training.trainer import Trainer
from meta_learning.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg = TrainRunConfig(**OmegaConf.to_container(cfg.run_cfg, resolve=True))  # type: ignore

    datasets_dir = os.environ.get("DATASETS_DIR") or str(cfg.data.base_dir)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    dataset_id = WHARDatasetID[run_cfg.dataset_id]
    dataset_cfg = get_dataset_cfg(dataset_id, datasets_dir)
    dataset_cfg.datasets_dir = str(datasets_dir)
    dataset_cfg.parallelize = bool(cfg.data.parallelize)
    dataset_cfg.selected_channels = (
        cfg.data.sensor_channels or dataset_cfg.selected_channels
    )

    experiment_id = run_cfg.create_experiment_id()
    tracker = create_tracker(
        backend=run_cfg.tracker_backend,
        experiment_name=experiment_id,
        tracking_uri=run_cfg.tracking_uri,
    )

    pre_pipeline = PreProcessingPipeline(dataset_cfg)
    _, session_df, window_df = pre_pipeline.run()

    splitter = LOSOSplitter(dataset_cfg)
    splits = splitter.get_splits(session_df, window_df)

    for split_idx, split in enumerate(splits):
        run_id = f"split_{split_idx}_{experiment_id}"
        logger.info("Running split %s / %s", split_idx, len(splits) - 1)

        post_pipeline = PostProcessingPipeline(
            dataset_cfg, pre_pipeline, window_df, split.train_indices
        )
        samples = post_pipeline.run()
        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        adapter = TorchAdapter(dataset_cfg, loader, split)
        dataloaders = adapter.get_dataloaders(batch_size=run_cfg.batch_size)

        model = TinyHAR(
            input_channels=len(
                dataset_cfg.selected_channels or dataset_cfg.available_channels
            ),
            window_size=int(dataset_cfg.window_time * dataset_cfg.sampling_freq),
            num_classes=dataset_cfg.num_of_activities,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

        trainer = Trainer(
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            test_loader=dataloaders["test"],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            tracker=tracker,
        )

        best_model_state, best_metrics = trainer.fit(run_name=run_id, run_cfg=run_cfg)

        split_dir = output_dir / run_id
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, split_dir / "state_dict.pth")

        params = run_cfg.to_tracking_dict()
        params["split_idx"] = split_idx

        with (split_dir / "params.json").open("w") as f:
            json.dump(params, f, indent=4)
        with (split_dir / "metrics.json").open("w") as f:
            json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    main()
