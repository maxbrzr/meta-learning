import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from whar_datasets import Loader
from whar_datasets.splitting.split import Split

from meta_learning.tracking import NullTracker, Tracker
from meta_learning.training.run_config import (
    MetaPretrainedRunConfig,
    MetaTrainRunConfig,
)
from meta_learning.utils.logging import get_logger

MetaRunConfig = MetaTrainRunConfig | MetaPretrainedRunConfig

logger = get_logger(__name__)


class EarlyStopping:
    """
    Implements early stopping logic.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class MetaTrainer:
    def __init__(
        self,
        loader: Loader,
        split: Split,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,  # <--- NEW: Needed to iterate classes
        shots_per_class: int | Tuple[int, int] = 5,
        batch_size: int = 32,
        num_train_batches_override: int | None = None,
        tracker: Tracker | None = None,
    ):
        """
        Args:
            loader: Your custom Loader instance.
            split: Split object containing train/val/test indices.
            model: The MetaTinyHAR model.
            optimizer: Optimizer.
            criterion: Loss function.
            device: 'cuda' or 'cpu'.
            shots_per_class:
                - int: fixed number of support samples per class
                - (min_shots, max_shots): sampled uniformly per batch/episode
            batch_size: Number of query samples per batch.
            num_train_batches_override:
                If set, use this exact number of training batches per epoch.
        """
        self.loader = loader
        self.split = split
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        if isinstance(shots_per_class, tuple):
            if len(shots_per_class) != 2:
                raise ValueError(
                    "shots_per_class range must be a tuple(min_shots, max_shots)."
                )
            min_shots, max_shots = shots_per_class
            if min_shots < 1 or max_shots < min_shots:
                raise ValueError(
                    "Invalid shots_per_class range. Expected min_shots >= 1 and max_shots >= min_shots."
                )
            self.shots_per_class_range: Tuple[int, int] | None = (min_shots, max_shots)
            self.shots_per_class_fixed: int | None = None
        else:
            if shots_per_class < 1:
                raise ValueError("shots_per_class must be >= 1.")
            self.shots_per_class_range = None
            self.shots_per_class_fixed = shots_per_class
        if num_train_batches_override is not None and num_train_batches_override < 1:
            raise ValueError("num_train_batches_override must be >= 1 when provided.")
        self.num_train_batches_override = num_train_batches_override
        self.batch_size = batch_size
        self.global_step = 0
        self.tracker = tracker or NullTracker()

    def _sample_shots_per_class(self) -> int:
        """
        Sample shots-per-class for the current batch/episode.
        The sampled value is shared across all samples in that batch, so support
        tensors remain rectangular (no masking needed).
        """
        if self.shots_per_class_range is not None:
            lo, hi = self.shots_per_class_range
            return int(np.random.randint(lo, hi + 1))
        assert self.shots_per_class_fixed is not None
        return self.shots_per_class_fixed

    def _eval_shots_per_class(self) -> int:
        """
        Deterministic shots-per-class for validation/test.
        If a range is configured, use its midpoint.
        """
        if self.shots_per_class_range is not None:
            lo, hi = self.shots_per_class_range
            return (lo + hi) // 2
        assert self.shots_per_class_fixed is not None
        return self.shots_per_class_fixed

    def _shot_sweep_values(self) -> List[int]:
        if self.shots_per_class_range is not None:
            lo, hi = self.shots_per_class_range
            return list(range(lo, hi + 1))
        assert self.shots_per_class_fixed is not None
        return [self.shots_per_class_fixed]

    def _load_query_batch(
        self, query_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load query tensors for explicit indices and move them to device.
        Returns (qx, qy) with shapes:
        - qx: (B, T, C)
        - qy: (B,)
        """
        query_acts: List[int] = []
        query_samples: List[List[np.ndarray]] = []
        for idx in query_indices:
            act, _, sample = self.loader.get_item(idx)
            query_acts.append(act)
            query_samples.append(sample)

        qx, qy = self._collate_tensors(query_samples, query_acts)
        return qx.to(self.device), qy.to(self.device)

    def _compute_support_context(
        self, sx: torch.Tensor, sy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute support context once and return:
        - task-level embedding z: (B, Z)
        - optional class embeddings: (B, N, H) for class-aware models
        """
        set_encoder_out = self.model.set_encoder(sx, sy)  # type: ignore[no-untyped-call]
        if hasattr(self.model.set_encoder, "to_task_embedding"):
            class_embeddings = set_encoder_out
            precomputed_z = self.model.set_encoder.to_task_embedding(class_embeddings)  # type: ignore[no-untyped-call]
            return precomputed_z, class_embeddings
        return set_encoder_out, None

    def _forward_with_context(
        self,
        qx: torch.Tensor,
        precomputed_z: torch.Tensor,
        class_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward query batch with precomputed support context.
        """
        batch_z = precomputed_z.expand(qx.shape[0], -1)
        if class_embeddings is not None:
            batch_class_embeddings = class_embeddings.expand(qx.shape[0], -1, -1)
            return self.model(
                x=qx,
                precomputed_z=batch_z,
                precomputed_class_embeddings=batch_class_embeddings,
            )
        return self.model(x=qx, precomputed_z=batch_z)

    def _aggregate_metrics(
        self,
        total_loss: float,
        total_examples: int,
        all_preds: List[torch.Tensor],
        all_targets: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Aggregate epoch-level loss/accuracy/macro-F1 from collected batches.
        """
        if total_examples == 0 or len(all_preds) == 0 or len(all_targets) == 0:
            # Can happen in strict calibration if all attempts are skipped.
            return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}

        avg_loss = total_loss / max(total_examples, 1)
        all_preds_np = torch.cat(all_preds).numpy()
        all_targets_np = torch.cat(all_targets).numpy()
        epoch_acc = (all_preds_np == all_targets_np).mean()
        epoch_f1 = float(
            f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
        )
        return {"loss": avg_loss, "accuracy": epoch_acc, "f1_macro": epoch_f1}

    def _collate_tensors(
        self, samples: List[List[np.ndarray]], labels: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts list of numpy samples to (Batch, Time, Channels) tensors.
        """
        batch_tensors = []
        for item in samples:
            # Stack sensor arrays along channel dimension (axis 1)
            # item is List[np.array(T, C_sub)]
            combined_sample = np.concatenate(item, axis=1)  # (T, C_total)
            batch_tensors.append(combined_sample)

        x_tensor = torch.tensor(np.array(batch_tensors), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(labels), dtype=torch.long)

        return x_tensor, y_tensor

    def _get_support_set(
        self, subject_id: int, available_indices: List[int], shots_per_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Samples K shots-per-class for a specific subject.
        Returns the support data (X and y) AND the exact indices used,
        so they can be excluded from the query set during evaluation.
        """
        support_x_list = []
        support_y_list = []
        all_selected_indices: List[int] = []

        # Iterate through every activity class to ensure perfect class balance
        for cls_id in range(self.num_classes):
            # Filter the pool for this specific subject and class
            target_indices = self.loader.filter_indices(
                indices=available_indices, subject_id=subject_id, activity_id=cls_id
            )

            if len(target_indices) == 0:
                raise ValueError(
                    f"No support samples for subject {subject_id}, class {cls_id} "
                    "in current split. Cannot form a complete support set."
                )

            # Sample K shots
            if len(target_indices) < shots_per_class:
                # If they don't have enough distinct windows, sample with replacement
                selected = np.random.choice(
                    target_indices, shots_per_class, replace=True
                ).tolist()
            else:
                # Standard sampling without replacement
                selected = np.random.choice(
                    target_indices, shots_per_class, replace=False
                ).tolist()

            # Keep track of exactly what we sampled
            all_selected_indices.extend(selected)

            # Retrieve the raw data for these specific indices
            acts, _, samples = self.loader.sample_items(len(selected), selected)

            # Convert to tensors
            sx, sy = self._collate_tensors(samples, acts)

            support_x_list.append(sx)
            support_y_list.append(sy)

        # Concatenate all classes together
        # Result shape: (Total_Samples, Time, Channels)
        # where Total_Samples = Num_Classes * shots_per_class
        final_sx = torch.cat(support_x_list, dim=0)
        final_sy = torch.cat(support_y_list, dim=0)

        # Permute to match the SetEncoder's expected input format
        # (Total_Support, T, C) -> (Total_Support, C, T)
        final_sx = final_sx.permute(0, 2, 1)

        return final_sx, final_sy, all_selected_indices

    def _generate_meta_batch(
        self,
        query_indices: List[int],
        support_pool_indices: List[int],
        shots_per_class: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates a batch of (Query, Support) pairs from explicit query indices.
        """
        # 1. Determine query subjects for per-item support sampling
        query_subjs: List[int] = []
        for idx in query_indices:
            _, subj, _ = self.loader.get_item(idx)
            query_subjs.append(subj)

        # Convert Query to Tensor: (Batch, Time, Channels)
        qx, qy = self._load_query_batch(query_indices)

        # 2. Sample Support Sets (One per query item)
        support_x_list = []
        support_y_list = []

        for q_idx, sid in zip(query_indices, query_subjs):
            # Prevent query-support leakage for this query item.
            support_pool_wo_query = [
                idx for idx in support_pool_indices if idx != q_idx
            ]
            sx, sy, _ = self._get_support_set(
                sid, support_pool_wo_query, shots_per_class=shots_per_class
            )
            support_x_list.append(sx)
            support_y_list.append(sy)

        # Stack support sets: (Batch, K, C, T)
        final_sx = torch.stack(support_x_list).to(self.device)
        final_sy = torch.stack(support_y_list).to(self.device)

        return {"x": qx, "y": qy, "sx": final_sx, "sy": final_sy}

    def _get_feasible_query_indices(self, indices_pool: List[int]) -> List[int]:
        """
        Returns query indices that can form leakage-free support sets where
        support samples are from the same subject and contain every class.
        """
        subject_labels = {
            idx: self.loader.get_subject_label(idx) for idx in indices_pool
        }
        activity_labels = {
            idx: self.loader.get_activity_label(idx) for idx in indices_pool
        }

        class_counts: Dict[Tuple[int, int], int] = {}
        for idx in indices_pool:
            key = (subject_labels[idx], activity_labels[idx])
            class_counts[key] = class_counts.get(key, 0) + 1

        feasible: List[int] = []
        for idx in indices_pool:
            sid = subject_labels[idx]
            q_act = activity_labels[idx]
            is_feasible = True
            for cls_id in range(self.num_classes):
                count = class_counts.get((sid, cls_id), 0)
                if cls_id == q_act:
                    count -= 1  # exclude the query itself to avoid leakage
                if count <= 0:
                    is_feasible = False
                    break
            if is_feasible:
                feasible.append(idx)
        return feasible

    def _run_epoch(
        self, indices: List[int], is_train: bool, desc: str
    ) -> Dict[str, float]:
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        total_examples = 0
        all_preds = []
        all_targets = []
        feasible_query_indices = self._get_feasible_query_indices(indices)

        if len(feasible_query_indices) == 0:
            raise ValueError(
                f"No feasible query indices for {desc}: cannot build same-subject, "
                "leakage-free support sets with full class coverage."
            )
        if not is_train and len(feasible_query_indices) != len(indices):
            logger.info(
                "%s: using %s/%s feasible query indices (strict same-subject, leakage-free support).",
                desc,
                len(feasible_query_indices),
                len(indices),
            )

        ordered_query_indices = list(feasible_query_indices)
        if is_train:
            if self.num_train_batches_override is not None:
                num_batches = self.num_train_batches_override
            else:
                num_batches = max(len(feasible_query_indices) // self.batch_size, 1)
        else:
            num_batches = math.ceil(len(ordered_query_indices) / self.batch_size)
        pbar = tqdm(range(num_batches), desc=desc, leave=False)

        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for batch_idx in pbar:
                shots_this_batch = (
                    self._sample_shots_per_class()
                    if is_train
                    else self._eval_shots_per_class()
                )
                if is_train:
                    self.optimizer.zero_grad()
                    # Training keeps stochastic query sampling from the train split.
                    sampled_query_indices = np.random.choice(
                        np.asarray(feasible_query_indices),
                        size=self.batch_size,
                        replace=True,
                    ).tolist()
                    batch_data = self._generate_meta_batch(
                        sampled_query_indices,
                        indices,
                        shots_per_class=shots_this_batch,
                    )
                else:
                    start = batch_idx * self.batch_size
                    end = min(
                        (batch_idx + 1) * self.batch_size,
                        len(ordered_query_indices),
                    )
                    query_batch_indices = ordered_query_indices[start:end]
                    batch_data = self._generate_meta_batch(
                        query_batch_indices,
                        indices,
                        shots_per_class=shots_this_batch,
                    )

                logits = self.model(
                    x=batch_data["x"],
                    support_x=batch_data["sx"],
                    support_y=batch_data["sy"],
                )
                loss = self.criterion(logits, batch_data["y"])

                if is_train:
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1

                # Batch Metrics
                preds = logits.argmax(dim=1)
                batch_loss = loss.item()
                batch_preds_cpu = preds.detach().cpu().numpy()
                batch_targets_cpu = batch_data["y"].detach().cpu().numpy()

                batch_acc = (batch_preds_cpu == batch_targets_cpu).mean()
                batch_f1 = float(
                    f1_score(
                        batch_targets_cpu,
                        batch_preds_cpu,
                        average="macro",
                        zero_division=0,
                    )
                )

                # Log Train Batch Metrics (The "High-Res" View)
                if is_train:
                    self.tracker.log_metrics(
                        {
                            f"{desc.lower()}/loss": batch_loss,
                            f"{desc.lower()}/accuracy": batch_acc,
                            f"{desc.lower()}/f1_macro": batch_f1,
                        },
                        step=self.global_step,
                    )

                batch_size_curr = int(batch_data["y"].shape[0])
                total_loss += batch_loss * batch_size_curr
                total_examples += batch_size_curr
                all_preds.append(torch.from_numpy(batch_preds_cpu))
                all_targets.append(torch.from_numpy(batch_targets_cpu))
                pbar.set_postfix(
                    {"loss": f"{batch_loss:.4f}", "accuracy": f"{batch_acc:.2f}"}
                )

        metrics = self._aggregate_metrics(
            total_loss=total_loss,
            total_examples=total_examples,
            all_preds=all_preds,
            all_targets=all_targets,
        )

        if not is_train:
            self.tracker.log_metrics(
                {
                    f"{desc.lower()}/loss": metrics["loss"],
                    f"{desc.lower()}/accuracy": metrics["accuracy"],
                    f"{desc.lower()}/f1_macro": metrics["f1_macro"],
                },
                step=self.global_step,
            )

        return metrics

    def _evaluate_calibration(
        self,
        indices_pool: List[int],
        desc: str,
        n_eval_episodes: int = 128,
        shots_per_class_override: int | None = None,
        log_to_tracker: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluates the model by simulating a real-world calibration phase.
        For each subject:
        1. Samples K shots per class (Support).
        2. Evaluates on ALL remaining data for that subject (Query).
        3. Repeats this N times and averages to reduce variance.
        """
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        all_preds = []
        all_targets = []
        skipped_missing_support = 0
        skipped_empty_query = 0

        # 1. Group all available indices by subject
        subject_to_indices: Dict[int, List[int]] = {}
        for idx in indices_pool:
            sid = self.loader.get_subject_label(idx)
            if sid not in subject_to_indices:
                subject_to_indices[sid] = []
            subject_to_indices[sid].append(idx)

        with torch.no_grad():
            pbar = tqdm(subject_to_indices.items(), desc=desc, leave=False)

            for subject_id, subj_indices in pbar:
                # Repeat the calibration process N times per subject
                for episode in range(n_eval_episodes):
                    shots_this_episode = (
                        shots_per_class_override
                        if shots_per_class_override is not None
                        else self._eval_shots_per_class()
                    )
                    try:
                        # a. Sample the fixed Support Set
                        sx, sy, support_idx = self._get_support_set(
                            subject_id,
                            subj_indices,
                            shots_per_class=shots_this_episode,
                        )

                        # Add Batch dimension: (1, K_total, C, T)
                        sx = sx.to(self.device).unsqueeze(0)
                        sy = sy.to(self.device).unsqueeze(0)

                    except ValueError:
                        # Skip if subject doesn't have enough classes to form a support set
                        skipped_missing_support += 1
                        logger.info(
                            "%s: skip subject %s, episode %s/%s "
                            "(cannot form full class-balanced support set for K=%s).",
                            desc,
                            subject_id,
                            episode + 1,
                            n_eval_episodes,
                            shots_this_episode,
                        )
                        continue

                    # b. The Query Set is EVERYTHING ELSE from this subject
                    query_idx = list(set(subj_indices) - set(support_idx))
                    if len(query_idx) == 0:
                        skipped_empty_query += 1
                        logger.info(
                            "%s: skip subject %s, episode %s/%s "
                            "(no query samples left after support selection).",
                            desc,
                            subject_id,
                            episode + 1,
                            n_eval_episodes,
                        )
                        continue

                    # c. Compute support context once per subject/episode.
                    precomputed_z, class_embeddings = self._compute_support_context(
                        sx, sy
                    )

                    # d. Process the query set in standard batches
                    for i in range(0, len(query_idx), self.batch_size):
                        batch_q_idx = query_idx[i : i + self.batch_size]

                        qx, qy = self._load_query_batch(batch_q_idx)
                        logits = self._forward_with_context(
                            qx=qx,
                            precomputed_z=precomputed_z,
                            class_embeddings=class_embeddings,
                        )

                        loss = self.criterion(logits, qy)

                        # Store metrics
                        preds = logits.argmax(dim=1)
                        total_loss += loss.item() * qy.shape[0]
                        total_examples += qy.shape[0]
                        all_preds.append(preds.cpu())
                        all_targets.append(qy.cpu())

        metrics = self._aggregate_metrics(
            total_loss=total_loss,
            total_examples=total_examples,
            all_preds=all_preds,
            all_targets=all_targets,
        )
        if skipped_missing_support > 0 or skipped_empty_query > 0:
            logger.info(
                "%s: calibration skips summary -> missing_support=%s, empty_query=%s",
                desc,
                skipped_missing_support,
                skipped_empty_query,
            )

        if log_to_tracker:
            self.tracker.log_metrics(
                {
                    f"{desc.lower()}/loss": metrics["loss"],
                    f"{desc.lower()}/accuracy": metrics["accuracy"],
                    f"{desc.lower()}/f1_macro": metrics["f1_macro"],
                },
                step=self.global_step,
            )

        return metrics

    def final_shot_sweep_evaluation(
        self, final_eval_episodes: int
    ) -> Dict[str, object]:
        """
        Run final calibration evaluation on the current model state.
        Sweeps over all shot values in the configured range (or one fixed value).
        """
        shot_values = self._shot_sweep_values()
        final_eval: Dict[str, Any] = {
            "episodes": final_eval_episodes,
            "test": {},
        }

        logger.info(
            "Running final evaluation on current model: episodes=%s, shots=%s",
            final_eval_episodes,
            shot_values,
        )

        for shots in shot_values:
            test_metrics = self._evaluate_calibration(
                self.split.test_indices,
                desc=f"Final-Test-K{shots}",
                n_eval_episodes=final_eval_episodes,
                shots_per_class_override=shots,
                log_to_tracker=False,
            )
            final_eval["test"][str(shots)] = test_metrics

        return final_eval

    def get_shot_sweep_values(self) -> List[int]:
        return self._shot_sweep_values()

    def evaluate_final_for_shot(
        self, shots_per_class: int, final_eval_episodes: int
    ) -> Dict[str, float]:
        return self._evaluate_calibration(
            self.split.test_indices,
            desc=f"Final-Test-K{shots_per_class}",
            n_eval_episodes=final_eval_episodes,
            shots_per_class_override=shots_per_class,
            log_to_tracker=False,
        )

    def fit(
        self,
        run_id: str,
        run_cfg: MetaRunConfig,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
        epochs = run_cfg.epochs
        patience = run_cfg.patience
        early_stopper = EarlyStopping(patience=patience)
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_metrics: Dict[str, Dict[str, Any]] = {}

        logger.info("Starting Meta-Training on %s for %s epochs.", self.device, epochs)
        if self.shots_per_class_range is not None:
            logger.info(
                "Config: Shots per Class range=%s "
                "(train sampled per batch, val/test fixed at midpoint=%s)",
                self.shots_per_class_range,
                self._eval_shots_per_class(),
            )
        else:
            logger.info("Config: Shots per Class=%s", self.shots_per_class_fixed)

        with self.tracker.start_run(run_name=run_id):
            params = run_cfg.to_tracking_dict()
            params["optimizer"] = type(self.optimizer).__name__
            params["model_class"] = type(self.model).__name__
            self.tracker.log_params(params)

            for epoch in range(epochs):
                logger.info("Epoch %s/%s", epoch + 1, epochs)

                # 1. Train
                train_metrics = self._run_epoch(
                    self.split.train_indices, is_train=True, desc="1-Train"
                )

                # 2. Validate
                val_metrics = self._evaluate_calibration(
                    self.split.val_indices, n_eval_episodes=8, desc="2-Val"
                )

                # 3. Test
                # test_metrics = self._evaluate_calibration(
                #     self.split.test_indices, n_eval_episodes=32, desc="3-Test"
                # )

                logger.info(
                    "Train: Loss %.4f | Acc %.4f | F1 %.4f | "
                    "Val: Loss %.4f | Acc %.4f | F1 %.4f",
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    train_metrics["f1_macro"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["f1_macro"],
                )

                # 4. Early Stopping
                if early_stopper(val_metrics["loss"]):
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_metrics = {
                        "train": train_metrics,
                        "val": val_metrics,
                        # "test": test_metrics,
                    }
                    logger.info("Best model saved.")

                if early_stopper.early_stop:
                    logger.info("Early stopping triggered after %s epochs.", epoch + 1)
                    break

        logger.info("Restoring best model weights...")
        self.model.load_state_dict(best_model_state)

        return best_model_state, best_metrics
