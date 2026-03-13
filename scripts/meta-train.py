import json
import os

import torch
from whar_datasets import (
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
    get_dataset_cfg,
)
from whar_datasets.splitting.split import Split

from meta_learning.lora.meta_tinyhar import MetaTinyHAR
from meta_learning.tracking import Tracker, create_tracker
from meta_learning.training.run_config import MetaTrainRunConfig
from meta_learning.training.meta_trainer import MetaTrainer
from meta_learning.utils.logging import get_logger

logger = get_logger(__name__)


def run(
    experiment_id: str,
    run_id: str,
    split_idx: int,
    split: Split,
    run_cfg: MetaTrainRunConfig,
    tracker: Tracker,
):
    # create and run post-processing pipeline for the specific split
    post_pipeline = PostProcessingPipeline(
        cfg, pre_pipeline, window_df, split.train_indices
    )
    samples = post_pipeline.run()

    # create dataloaders for the specific split
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

    logger.info("num subjects / splits: %s/%s", cfg.num_of_subjects, len(splits))
    logger.info("num channels: %s", len(cfg.sensor_channels))
    logger.info("Number of training indices: %s", len(split.train_indices))
    logger.info("Number of validation indices: %s", len(split.val_indices))
    logger.info("Number of test indices: %s", len(split.test_indices))

    model = MetaTinyHAR(
        input_channels=len(cfg.sensor_channels),
        window_size=int(cfg.window_time * cfg.sampling_freq),
        num_classes=cfg.num_of_activities,
        set_encoder_variant=run_cfg.set_encoder_variant,
        set_encoder_num_heads=run_cfg.set_encoder_num_heads,
        class_aware=run_cfg.class_aware,
        hypernetwork_variant=run_cfg.hypernetwork_variant,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    trainer = MetaTrainer(
        loader=loader,
        split=split,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=len(cfg.activity_names),
        shots_per_class=run_cfg.shots_per_class,
        batch_size=run_cfg.batch_size,
        num_train_batches_override=run_cfg.num_train_batches_override,
        tracker=tracker,
    )

    # Defensive cleanup for notebook/restart scenarios where a run is still open.
    tracker.end_active_runs()

    best_model_state, best_metrics = trainer.fit(
        run_id=run_id,
        run_cfg=run_cfg,
    )

    root_dir = "./results"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    exp_dir = f"{root_dir}/{experiment_id}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    run_dir = f"{exp_dir}/{run_id}"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    torch.save(best_model_state, f"{run_dir}/state_dict.pth")
    params = run_cfg.to_tracking_dict()
    params["split_idx"] = split_idx
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Final sweep evaluation on restored best model.
    # Persist metrics after each K-shot result to avoid losing progress on long sweeps.
    best_metrics["final_eval"] = {"episodes": run_cfg.final_eval_episodes, "test": {}}
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)

    for shots in trainer.get_shot_sweep_values():
        test_metrics = trainer.evaluate_final_for_shot(
            shots_per_class=shots,
            final_eval_episodes=run_cfg.final_eval_episodes,
        )
        best_metrics["final_eval"]["test"][str(shots)] = test_metrics  # type: ignore[index]
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=4)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    dataset_id = WHARDatasetID.UCI_HAR
    run_cfg = MetaTrainRunConfig(dataset_id=dataset_id.name)
    experiment_id = run_cfg.create_experiment_id()

    tracker = create_tracker(
        backend=run_cfg.tracker_backend,  # type: ignore[arg-type]
        experiment_name=experiment_id,
        tracking_uri=run_cfg.tracking_uri,
    )

    # create cfg for UCI HAR dataset
    cfg = get_dataset_cfg(dataset_id, "./datasets")
    cfg.parallelize = True

    # create and run pre-processing pipeline
    pre_pipeline = PreProcessingPipeline(cfg)
    activity_df, session_df, window_df = pre_pipeline.run()

    # create LOSO splits
    splitter = LOSOSplitter(cfg)
    splits = splitter.get_splits(session_df, window_df)

    for i in range(len(splits)):
        split = splits[i]
        logger.info("Running split %s / %s", i, len(splits) - 1)
        run_id = f"split_{i}_{experiment_id}"
        run(
            experiment_id,
            run_id,
            i,
            split,
            run_cfg,
            tracker,
        )
