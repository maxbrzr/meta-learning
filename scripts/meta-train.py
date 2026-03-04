import json
import os

import mlflow
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
from meta_learning.training.meta_trainer import MetaTrainer


def run(
    experiment_id: str,
    run_id: str,
    split: Split,
    num_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    shots_per_class: int,
):
    # create and run post-processing pipeline for the specific split
    post_pipeline = PostProcessingPipeline(
        cfg, pre_pipeline, window_df, split.train_indices
    )
    samples = post_pipeline.run()

    # create dataloaders for the specific split
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

    print(f"num subjects / splits: {cfg.num_of_subjects}/{len(splits)}")
    print(f"num channels: {len(cfg.sensor_channels)}")
    print(f"Number of training indices: {len(split.train_indices)}")
    print(f"Number of validation indices: {len(split.val_indices)}")
    print(f"Number of test indices: {len(split.test_indices)}")

    model = MetaTinyHAR(
        input_channels=len(cfg.sensor_channels),
        window_size=int(cfg.window_time * cfg.sampling_freq),
        num_classes=cfg.num_of_activities,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        shots_per_class=shots_per_class,
        batch_size=batch_size,
    )

    best_model_state, best_metrics = trainer.fit(
        run_id=run_id, epochs=num_epochs, patience=patience
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
    num_epochs = 20
    patience = 5
    batch_size = 16
    learning_rate = 0.0003
    shots_per_class = 10

    dataset_id = WHARDatasetID.UCI_HAR
    experiment_id = f"{dataset_id.name.lower()}_lora_meta_tinyhar_ep{num_epochs}_pat{patience}_bs{batch_size}_lr{learning_rate}_shots{shots_per_class}"

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(experiment_id)

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
        print(f"Running split {i} / {len(splits) - 1}")
        run_id = f"split_{i}_{experiment_id}"
        run(
            experiment_id,
            run_id,
            split,
            num_epochs,
            patience,
            batch_size,
            learning_rate,
            shots_per_class,
        )
