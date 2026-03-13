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
    WHARDatasetID,
    get_dataset_cfg,
)

import hydra
from meta_learning.lora.meta_tinyhar import MetaTinyHAR
from meta_learning.tracking import create_tracker
from meta_learning.training.meta_trainer import MetaTrainer
from meta_learning.training.run_config import MetaTrainRunConfig
from meta_learning.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg = MetaTrainRunConfig(**OmegaConf.to_container(cfg.run_cfg, resolve=True))  # type: ignore
    dataset_id = WHARDatasetID[run_cfg.dataset_id]

    datasets_dir = os.environ.get("DATASETS_DIR") or str(cfg.data.base_dir)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    dataset_cfg = get_dataset_cfg(dataset_id, datasets_dir)
    dataset_cfg.datasets_dir = str(datasets_dir)
    if cfg.data.sensor_channels is not None:
        dataset_cfg.sensor_channels = cfg.data.sensor_channels
    dataset_cfg.parallelize = bool(cfg.data.parallelize)

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

        model = MetaTinyHAR(
            input_channels=len(dataset_cfg.sensor_channels),
            window_size=int(dataset_cfg.window_time * dataset_cfg.sampling_freq),
            num_classes=dataset_cfg.num_of_activities,
            set_encoder_variant=run_cfg.set_encoder_variant,
            set_encoder_num_heads=run_cfg.set_encoder_num_heads,
            class_aware=run_cfg.class_aware,
            hypernetwork_variant=run_cfg.hypernetwork_variant,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

        trainer = MetaTrainer(
            loader=loader,
            split=split,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=len(dataset_cfg.activity_names),
            shots_per_class=run_cfg.shots_per_class,
            batch_size=run_cfg.batch_size,
            num_train_batches_override=run_cfg.num_train_batches_override,
            tracker=tracker,
        )

        tracker.end_active_runs()
        best_model_state, best_metrics = trainer.fit(run_id=run_id, run_cfg=run_cfg)

        split_dir = output_dir / run_id
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, split_dir / "state_dict.pth")

        params = run_cfg.to_tracking_dict()
        params["split_idx"] = split_idx

        with (split_dir / "params.json").open("w") as f:
            json.dump(params, f, indent=4)

        best_metrics["final_eval"] = {
            "episodes": run_cfg.final_eval_episodes,
            "test": {},
        }
        with (split_dir / "metrics.json").open("w") as f:
            json.dump(best_metrics, f, indent=4)

        for shots in trainer.get_shot_sweep_values():
            test_metrics = trainer.evaluate_final_for_shot(
                shots_per_class=shots,
                final_eval_episodes=run_cfg.final_eval_episodes,
            )
            best_metrics["final_eval"]["test"][str(shots)] = test_metrics
            with (split_dir / "metrics.json").open("w") as f:
                json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    main()
