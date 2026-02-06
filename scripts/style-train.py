# import json
# import os

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

from meta_learning.models.tiny_har import TinyHAR
from meta_learning.style.dual_head_set_classifier import DualHeadSetClassifier
from meta_learning.style.dual_head_trainer import DualHeadTrainer
from meta_learning.style.set_encoder import (
    BayesianSetEncoder,
    MeanAttentiveSetEncoder,
    MeanSetEncoder,
    QueryAttentiveSetEncoder,
)


def run(
    experiment_id: str,
    run_id: str,
    split: Split,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    feature_dim: int,
    context_size: int,
    kl_weight: float,
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

    encoder = TinyHAR(
        input_channels=len(cfg.sensor_channels),
        window_size=int(cfg.window_time * cfg.sampling_freq),
        num_classes=cfg.num_of_activities,
        num_filters=feature_dim // 2,
    )

    # set_encoder = MeanSetEncoder(
    #     encoder=encoder,
    #     feature_dim=feature_dim,
    # )

    set_encoder = QueryAttentiveSetEncoder(
        encoder=encoder,
        feature_dim=feature_dim,
    )

    model = DualHeadSetClassifier(
        set_encoder=set_encoder,
        feature_dim=feature_dim,
        num_subjects=cfg.num_of_subjects,
        num_activities=len(cfg.activity_names),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    trainer = DualHeadTrainer(
        loader=loader,
        split=split,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_subjects=cfg.num_of_subjects,
        num_activities=len(cfg.activity_names),
        batch_size=batch_size,
        context_size=context_size,
        kl_weight=kl_weight,
    )

    trainer.fit(run_id=run_id, epochs=num_epochs)
    # best_model_state, best_metrics = trainer.fit(run_id=run_id, epochs=num_epochs)

    # root_dir = "./results"
    # if os.path.exists(root_dir) is False:
    #     os.mkdir(root_dir)

    # sub_dir = experiment_id
    # if os.path.exists(f"{root_dir}/{sub_dir}") is False:
    #     os.mkdir(f"{root_dir}/{sub_dir}")

    # sub_sub_dir = run_id
    # if os.path.exists(f"{root_dir}/{sub_sub_dir}") is False:
    #     os.mkdir(f"{root_dir}/{sub_sub_dir}")

    # torch.save(best_model_state, f"{root_dir}/{sub_sub_dir}/state_dict.pth")
    # with open(f"{root_dir}/{sub_sub_dir}/metrics.json", "w") as f:
    #     json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    num_epochs = 60
    batch_size = 32
    learning_rate = 0.001
    feature_dim = 64
    context_size = 10
    kl_weight = 0.1

    dataset_id = WHARDatasetID.UCI_HAR
    experiment_id = f"styletinyhar-{dataset_id.name.lower()}_ep{num_epochs}_bs{batch_size}_lr{learning_rate}_context{context_size}_kl{kl_weight}"

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
        if i > 0:
            break  # for testing, only run on first split
        split = splits[i]
        print(f"Running split {i} / {len(splits) - 1}")
        run_id = f"split{i}_{experiment_id}"
        run(
            experiment_id,
            run_id,
            split,
            num_epochs,
            batch_size,
            learning_rate,
            feature_dim,
            context_size,
            kl_weight,
        )
