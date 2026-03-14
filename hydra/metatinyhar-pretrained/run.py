import json
import os
from glob import glob
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
from meta_learning.training.run_config import MetaPretrainedRunConfig
from meta_learning.utils.logging import get_logger

logger = get_logger(__name__)


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
            return checkpoint
    raise ValueError(f"Unsupported checkpoint format at: {path}")


def _resolve_tinyhar_checkpoint_path(
    split_idx: int,
    dataset_name: str,
    checkpoint_path_or_dir: str | None,
    results_root: str,
) -> str:
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
    candidates = [
        p for p in candidates if "lora" not in p.lower() and "meta" not in p.lower()
    ]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not resolve TinyHAR checkpoint for split {split_idx} in '{results_root}'. "
        f"Searched pattern: {pattern}"
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg_dict = OmegaConf.to_container(cfg.run_cfg, resolve=True)
    run_cfg = MetaPretrainedRunConfig(**run_cfg_dict)  # type: ignore[arg-type]

    override_ckpt = os.getenv("TINYHAR_CHECKPOINT_PATH_OVERRIDE")
    if override_ckpt:
        run_cfg.tinyhar_checkpoint_path = override_ckpt

    datasets_dir = os.environ.get("DATASETS_DIR") or str(cfg.data.base_dir)
    pretrained_results_dir = os.environ.get("PRETRAINED_RESULTS_DIR") or str(
        cfg.paths.pretrained_results_dir
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    dataset_id: WHARDatasetID = WHARDatasetID[run_cfg.dataset_id]
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

        model = MetaTinyHAR(
            input_channels=len(
                dataset_cfg.selected_channels or dataset_cfg.available_channels
            ),
            window_size=int(dataset_cfg.window_time * dataset_cfg.sampling_freq),
            num_classes=dataset_cfg.num_of_activities,
            set_encoder_variant=run_cfg.set_encoder_variant,
            set_encoder_num_heads=run_cfg.set_encoder_num_heads,
            class_aware=run_cfg.class_aware,
            hypernetwork_variant=run_cfg.hypernetwork_variant,
        )

        checkpoint_path = _resolve_tinyhar_checkpoint_path(
            split_idx=split_idx,
            dataset_name=run_cfg.dataset_id,
            checkpoint_path_or_dir=(
                os.path.abspath(run_cfg.tinyhar_checkpoint_path)
                if run_cfg.tinyhar_checkpoint_path
                else None
            ),
            results_root=pretrained_results_dir,
        )
        logger.info("Using TinyHAR checkpoint: %s", checkpoint_path)
        tinyhar_state_dict = _load_tinyhar_state_dict(checkpoint_path)
        load_stats = model.load_pretrained_tinyhar(tinyhar_state_dict)
        logger.info(
            "Loaded TinyHAR pretrained weights: %s tensors (skipped: %s).",
            load_stats["loaded"],
            load_stats["skipped"],
        )

        model.freeze_for_meta_learning()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            "Trainable parameter tensors after freezing: %s", len(trainable_params)
        )

        optimizer = torch.optim.Adam(trainable_params, lr=run_cfg.learning_rate)
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
            num_classes=len(
                dataset_cfg.selected_channels or dataset_cfg.available_channels
            ),
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
        params["resolved_tinyhar_checkpoint_path"] = checkpoint_path
        params["pretrained_results_dir"] = pretrained_results_dir

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
