import json
import os
from glob import glob

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
from meta_learning.training.run_config import MetaPretrainedRunConfig
from meta_learning.training.meta_trainer import MetaTrainer


def _load_tinyhar_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(
            checkpoint["model_state_dict"], dict
        ):
            return checkpoint["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint  # already a plain state dict
    raise ValueError(f"Unsupported checkpoint format at: {path}")


def _resolve_tinyhar_checkpoint_path(
    split_idx: int,
    dataset_name: str,
    checkpoint_path_or_dir: str | None = None,
    results_root: str = "./results",
) -> str:
    """
    Resolve the correct vanilla TinyHAR checkpoint for the given split.
    Priority:
    1) explicit file path
    2) explicit directory containing matching split checkpoint
    3) auto-discovery in results/ using dataset + split index
    """
    if checkpoint_path_or_dir:
        if os.path.isfile(checkpoint_path_or_dir):
            split_tag = f"split_{split_idx}_"
            if split_tag not in checkpoint_path_or_dir:
                raise ValueError(
                    "Explicit checkpoint file does not match current split. "
                    f"Expected '{split_tag}' in path, got: {checkpoint_path_or_dir}"
                )
            return checkpoint_path_or_dir

        if os.path.isdir(checkpoint_path_or_dir):
            dir_candidates = sorted(
                glob(
                    os.path.join(
                        checkpoint_path_or_dir, f"split_{split_idx}_*", "state_dict.pth"
                    )
                )
            )
            if dir_candidates:
                return dir_candidates[0]

    dataset_prefix = dataset_name.lower()
    pattern = os.path.join(
        results_root,
        f"{dataset_prefix}_tinyhar*",
        f"split_{split_idx}_*",
        "state_dict.pth",
    )
    candidates = sorted(glob(pattern))

    # Keep only vanilla TinyHAR experiments (exclude meta/lora variants)
    candidates = [
        p for p in candidates if "lora" not in p.lower() and "meta" not in p.lower()
    ]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not resolve TinyHAR checkpoint for split {split_idx} in '{results_root}'. "
        f"Searched pattern: {pattern}"
    )


def run(
    experiment_id: str,
    run_id: str,
    split_idx: int,
    split: Split,
    run_cfg: MetaPretrainedRunConfig,
    tracker: Tracker,
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
        set_encoder_variant=run_cfg.set_encoder_variant,
        set_encoder_num_heads=run_cfg.set_encoder_num_heads,
        class_aware=run_cfg.class_aware,
        hypernetwork_variant=run_cfg.hypernetwork_variant,
    )

    resolved_checkpoint_path = _resolve_tinyhar_checkpoint_path(
        split_idx=split_idx,
        dataset_name=dataset_id.name,
        checkpoint_path_or_dir=run_cfg.tinyhar_checkpoint_path,
        results_root="./results",
    )
    print(f"Using TinyHAR checkpoint: {resolved_checkpoint_path}")
    tinyhar_state_dict = _load_tinyhar_state_dict(resolved_checkpoint_path)
    load_stats = model.load_pretrained_tinyhar(tinyhar_state_dict)
    print(
        f"Loaded TinyHAR pretrained weights: {load_stats['loaded']} tensors "
        f"(skipped: {load_stats['skipped']})."
    )

    model.freeze_for_meta_learning()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors after freezing: {len(trainable_params)}")

    optimizer = torch.optim.Adam(trainable_params, lr=run_cfg.learning_rate)
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
    dataset_id = WHARDatasetID.DSADS
    run_cfg = MetaPretrainedRunConfig(dataset_id=dataset_id.name)

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
            i,
            split,
            run_cfg,
            tracker,
        )
