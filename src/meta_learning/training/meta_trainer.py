import copy
import math
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from whar_datasets import Loader
from whar_datasets.splitting.split import Split


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
        shots_per_class: int = 5,  # <--- CHANGED: Now means "Shots PER CLASS"
        batch_size: int = 32,
    ):
        """
        Args:
            loader: Your custom Loader instance.
            split: Split object containing train/val/test indices.
            model: The MetaTinyHAR model.
            optimizer: Optimizer.
            criterion: Loss function.
            device: 'cuda' or 'cpu'.
            shots_per_class: Number of support samples per class to generate.
            batch_size: Number of query samples per batch.
        """
        self.loader = loader
        self.split = split
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.shots_per_class = shots_per_class
        self.batch_size = batch_size
        self.global_step = 0

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
        self, subject_id: int, available_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples K shots PER CLASS for a specific subject.
        Support samples are always from the same subject.
        """
        support_x_list = []
        support_y_list = []

        # Iterate through every activity class to ensure balance
        for cls_id in range(self.num_classes):
            subject_class_indices = self.loader.filter_indices(
                indices=available_indices, subject_id=subject_id, activity_id=cls_id
            )
            target_indices = subject_class_indices

            if len(target_indices) == 0:
                raise ValueError(
                    f"No support samples for subject {subject_id}, class {cls_id} "
                    "in current split after leakage exclusion."
                )

            # Sample K shots
            if len(target_indices) < self.shots_per_class:
                # If they only have 1 sample but we want 2, duplicate it
                selected = np.random.choice(
                    target_indices, self.shots_per_class, replace=True
                ).tolist()
            else:
                selected = np.random.choice(
                    target_indices, self.shots_per_class, replace=False
                ).tolist()

            # Retrieve Data
            acts, _, samples = self.loader.sample_items(len(selected), selected)
            sx, sy = self._collate_tensors(samples, acts)

            support_x_list.append(sx)
            support_y_list.append(sy)

        # Concatenate all classes
        # Result shape: (Total_Samples, Time, Channels)
        # where Total_Samples = Num_Present_Classes * K_Shots
        final_sx = torch.cat(support_x_list, dim=0)
        final_sy = torch.cat(support_y_list, dim=0)

        # Permute for SetEncoder: (Total_Support, T, C) -> (Total_Support, C, T)
        final_sx = final_sx.permute(0, 2, 1)

        return final_sx, final_sy

    def _generate_meta_batch(
        self, query_indices: List[int], support_pool_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Generates a batch of (Query, Support) pairs from explicit query indices.
        """
        # 1. Deterministically fetch query samples from given indices
        query_acts: List[int] = []
        query_subjs: List[int] = []
        query_samples = []
        for idx in query_indices:
            act, subj, sample = self.loader.get_item(idx)
            query_acts.append(act)
            query_subjs.append(subj)
            query_samples.append(sample)

        # Convert Query to Tensor: (Batch, Time, Channels)
        qx, qy = self._collate_tensors(query_samples, query_acts)
        qx, qy = qx.to(self.device), qy.to(self.device)

        # 2. Sample Support Sets (One per query item)
        support_x_list = []
        support_y_list = []

        for q_idx, sid in zip(query_indices, query_subjs):
            # Prevent query-support leakage for this query item.
            support_pool_wo_query = [
                idx for idx in support_pool_indices if idx != q_idx
            ]
            sx, sy = self._get_support_set(sid, support_pool_wo_query)
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
        if is_train:
            self.model.train()
        else:
            self.model.eval()

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
            print(
                f"{desc}: using {len(feasible_query_indices)}/{len(indices)} feasible "
                "query indices (strict same-subject, leakage-free support)."
            )

        if is_train:
            num_batches = len(feasible_query_indices) // self.batch_size
            if num_batches == 0:
                num_batches = 1
            pbar = tqdm(range(num_batches), desc=desc, leave=False)
        else:
            ordered_query_indices = list(feasible_query_indices)
            num_batches = math.ceil(len(ordered_query_indices) / self.batch_size)
            pbar = tqdm(range(num_batches), desc=desc, leave=False)

        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for batch_idx in pbar:
                if is_train:
                    self.optimizer.zero_grad()
                    # Training keeps stochastic query sampling from the train split.
                    sampled_query_indices = np.random.choice(
                        np.asarray(feasible_query_indices),
                        size=self.batch_size,
                        replace=True,
                    ).tolist()
                    batch_data = self._generate_meta_batch(
                        sampled_query_indices, indices
                    )
                else:
                    start = batch_idx * self.batch_size
                    end = min(
                        (batch_idx + 1) * self.batch_size,
                        len(ordered_query_indices),  # type: ignore
                    )
                    query_batch_indices = ordered_query_indices[start:end]  # type: ignore
                    batch_data = self._generate_meta_batch(query_batch_indices, indices)

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
                    mlflow.log_metrics(  # type: ignore
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

        # Aggregation
        avg_loss = total_loss / max(total_examples, 1)
        all_preds_np = torch.cat(all_preds).numpy()
        all_targets_np = torch.cat(all_targets).numpy()
        epoch_acc = (all_preds_np == all_targets_np).mean()
        epoch_f1 = float(
            f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
        )

        if not is_train:
            mlflow.log_metrics(  # type: ignore
                {
                    f"{desc.lower()}/loss": avg_loss,
                    f"{desc.lower()}/accuracy": epoch_acc,
                    f"{desc.lower()}/f1_macro": epoch_f1,
                },
                step=self.global_step,
            )

        return {"loss": avg_loss, "accuracy": epoch_acc, "f1_macro": epoch_f1}

    def fit(
        self, run_id: str, epochs: int = 20, patience: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        early_stopper = EarlyStopping(patience=patience)
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_metrics: Dict[str, Dict[str, float]] = {}

        print(f"Starting Meta-Training on {self.device} for {epochs} epochs.")
        print(f"Config: Shots per Class={self.shots_per_class}")

        with mlflow.start_run(run_name=run_id):  # type : ignore
            mlflow.log_params(  # type : ignore
                {
                    "epochs": epochs,
                    "patience": patience,
                    "batch_size": self.batch_size,
                    "shots_per_class": self.shots_per_class,
                    "optimizer": type(self.optimizer).__name__,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                # 1. Train
                train_metrics = self._run_epoch(
                    self.split.train_indices, is_train=True, desc="1-Train"
                )

                # 2. Validate
                val_metrics = self._run_epoch(
                    self.split.val_indices, is_train=False, desc="2-Val"
                )

                # 3. Test
                test_metrics = self._run_epoch(
                    self.split.test_indices, is_train=False, desc="3-Test"
                )

                print(
                    f"Train: Loss {train_metrics['loss']:.4f} | Acc {train_metrics['accuracy']:.4f} | F1 {train_metrics['f1_macro']:.4f}\n"
                    f"Val:   Loss {val_metrics['loss']:.4f} | Acc {val_metrics['accuracy']:.4f} | F1 {val_metrics['f1_macro']:.4f}\n"
                    f"Test:  Loss {test_metrics['loss']:.4f} | Acc {test_metrics['accuracy']:.4f} | F1 {test_metrics['f1_macro']:.4f}"
                )

                # 4. Early Stopping
                if early_stopper(val_metrics["loss"]):
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_metrics = {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    }
                    print(">> Best model saved.")

                if early_stopper.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        print("Restoring best model weights...")
        self.model.load_state_dict(best_model_state)

        return best_model_state, best_metrics
