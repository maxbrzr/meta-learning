import json
import os

import torch
from whar_datasets import (
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)
from whar_datasets.splitting.split import Split

from meta_learning.models.tiny_har import TinyHAR
from meta_learning.tracking import Tracker, create_tracker
from meta_learning.training.run_config import TinyHARRunConfig
from meta_learning.training.trainer import Trainer


def run(
    experiment_id: str,
    run_id: str,
    split: Split,
    run_cfg: TinyHARRunConfig,
    tracker: Tracker,
):
    # create and run post-processing pipeline for the specific split
    post_pipeline = PostProcessingPipeline(
        cfg, pre_pipeline, window_df, split.train_indices
    )
    samples = post_pipeline.run()

    # create dataloaders for the specific split
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
    adapter = TorchAdapter(cfg, loader, split)
    dataloaders = adapter.get_dataloaders(batch_size=run_cfg.batch_size)
    train_loader, val_loader, test_loader = (
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
    )

    print(f"num subjects / splits: {cfg.num_of_subjects}/{len(splits)}")
    print(f"num channels: {len(cfg.sensor_channels)}")
    print(f"Number of training samples: {len(train_loader) * run_cfg.batch_size}")
    print(
        f"Number of validation samples: {len(val_loader) * run_cfg.batch_size}"
    )
    print(f"Number of test samples: {len(test_loader) * run_cfg.batch_size}")
    print(f"Number of training indices: {len(split.train_indices)}")
    print(f"Number of validation indices: {len(split.val_indices)}")
    print(f"Number of test indices: {len(split.test_indices)}")

    model = TinyHAR(
        input_channels=len(cfg.sensor_channels),
        window_size=int(cfg.window_time * cfg.sampling_freq),
        num_classes=cfg.num_of_activities,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        tracker=tracker,
    )

    best_model_state, best_metrics = trainer.fit(
        run_id,
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
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    dataset_id = WHARDatasetID.DSADS
    run_cfg = TinyHARRunConfig(dataset_id=dataset_id.name)

    # create cfg for UCI HAR dataset
    cfg = get_dataset_cfg(dataset_id, "./datasets")
    cfg.sensor_channels = [
        "RA_xacc",
        "RA_yacc",
        "RA_zacc",
        "RA_xgyro",
        "RA_ygyro",
        "RA_zgyro",
        "RL_xacc",
        "RL_yacc",
        "RL_zacc",
        "RL_xgyro",
        "RL_ygyro",
        "RL_zgyro",
    ]
    cfg.parallelize = True

    experiment_id = run_cfg.create_experiment_id()
    tracker = create_tracker(
        backend=run_cfg.tracker_backend,  # type: ignore[arg-type]
        experiment_name=experiment_id,
        tracking_uri=run_cfg.tracking_uri,
    )

    # create and run pre-processing pipeline
    pre_pipeline = PreProcessingPipeline(cfg)
    activity_df, session_df, window_df = pre_pipeline.run()

    # create LOSO splits
    splitter = LOSOSplitter(cfg)
    splits = splitter.get_splits(session_df, window_df)

    for i in range(len(splits)):
        split = splits[i]
        print(f"Running split {i} / {len(splits) - 1}")
        run_id = f"split_{i}_{experiment_id}"
        run(
            experiment_id,
            run_id,
            split,
            run_cfg,
            tracker,
        )
