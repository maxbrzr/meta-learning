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

# Assuming the model class is imported elsewhere
# from meta_learning.style.dual_head_set_classifier import DualHeadSetClassifier


class DualHeadTrainer:
    def __init__(
        self,
        loader: Loader,
        split: Split,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_subjects: int,
        num_activities: int,
        context_size: int = 5,
        batch_size: int = 32,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        kl_weight: float = 0.5,
    ):
        self.loader = loader
        self.split = split
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_subjects = num_subjects
        self.num_activities = num_activities
        self.context_size = context_size
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.global_step = 0

    def _generate_batch(
        self, pool_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_sets = []
        batch_subj_labels = []
        batch_act_labels = []

        while len(batch_sets) < self.batch_size:
            s_id = np.random.randint(0, self.num_subjects)
            a_id = np.random.randint(0, self.num_activities)

            try:
                _, _, samples = self.loader.sample_items(
                    batch_size=self.context_size,
                    indices=pool_indices,
                    subject_id=s_id,
                    activity_id=a_id,
                )
                samples_tensor = torch.stack([torch.tensor(s[0]) for s in samples])
                batch_sets.append(samples_tensor)
                batch_subj_labels.append(s_id)
                batch_act_labels.append(a_id)
            except (AssertionError, ValueError):
                continue

        x_tensor = torch.stack(batch_sets).to(torch.float32)
        y_subj = torch.tensor(batch_subj_labels, dtype=torch.long)
        y_act = torch.tensor(batch_act_labels, dtype=torch.long)

        return x_tensor, y_subj, y_act

    def _compute_kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kl.mean()

    def _run_epoch(
        self, indices: List[int], is_train: bool, desc: str
    ) -> Dict[str, float]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_kl = 0.0, 0.0
        num_batches = max(1, len(indices) // (self.batch_size * self.context_size))

        all_preds_task, all_targets_task = [], []
        all_preds_adv, all_targets_adv = [], []

        pbar = tqdm(range(num_batches), desc=desc, leave=False)
        context = torch.enable_grad() if is_train else torch.no_grad()
        log_prefix = desc.lower().replace("-", "_")

        with context:
            for _ in pbar:
                if is_train:
                    self.optimizer.zero_grad()

                bx, by_subj, by_act = self._generate_batch(indices)
                bx, by_subj, by_act = (
                    bx.to(self.device),
                    by_subj.to(self.device),
                    by_act.to(self.device),
                )

                logits_task, logits_adv, mu, logvar = self.model(bx)

                loss_task = self.criterion(logits_task, by_subj)
                loss_adv = self.criterion(logits_adv, by_act)
                loss_kl = self._compute_kl_loss(mu, logvar)

                # Combined Loss
                loss = loss_task + loss_adv + (self.kl_weight * loss_kl)

                if is_train:
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1

                # Batch Metrics
                preds_task = logits_task.argmax(dim=1).cpu()
                preds_adv = logits_adv.argmax(dim=1).cpu()
                batch_task_acc = (preds_task == by_subj.cpu()).numpy().mean()
                batch_task_f1 = float(
                    f1_score(
                        by_subj.cpu(), preds_task, average="macro", zero_division=0
                    )
                )
                batch_adv_acc = (preds_adv == by_act.cpu()).numpy().mean()
                batch_adv_f1 = float(
                    f1_score(by_act.cpu(), preds_adv, average="macro", zero_division=0)
                )

                # --- NEW: High-Res Logging for Training ---
                if is_train:
                    mlflow.log_metrics(  # type: ignore
                        {
                            f"{log_prefix}/loss": loss.item(),
                            f"{log_prefix}/loss_kl": loss_kl.item(),
                            f"{log_prefix}/task_acc": batch_task_acc,
                            f"{log_prefix}/task_f1": batch_task_f1,
                            f"{log_prefix}/adv_acc": batch_adv_acc,
                            f"{log_prefix}/adv_f1": batch_adv_f1,
                        },
                        step=self.global_step,
                    )

                total_loss += loss.item()
                total_kl += loss_kl.item()
                all_preds_task.append(preds_task)
                all_targets_task.append(by_subj.cpu())
                all_preds_adv.append(preds_adv)
                all_targets_adv.append(by_act.cpu())

                pbar.set_postfix(
                    {"L": f"{loss.item():.2f}", "KL": f"{loss_kl.item():.2f}"}
                )

        # Epoch-level Aggregation
        preds_task = torch.cat(all_preds_task)
        targets_task = torch.cat(all_targets_task)
        epoch_task_acc = (preds_task == targets_task).numpy().mean()
        epoch_task_f1 = f1_score(
            targets_task, preds_task, average="macro", zero_division=0
        )

        preds_adv = torch.cat(all_preds_adv)
        targets_adv = torch.cat(all_targets_adv)
        epoch_adv_acc = (preds_adv == targets_adv).numpy().mean()
        epoch_adv_f1 = f1_score(
            targets_adv, preds_adv, average="macro", zero_division=0
        )

        metrics = {
            "loss": total_loss / num_batches,
            "loss_kl": total_kl / num_batches,
            "task_acc": epoch_task_acc,
            "task_f1": epoch_task_f1,
            "adv_acc": epoch_adv_acc,
            "adv_f1": epoch_adv_f1,
        }

        # Log summary metrics for Validation/Test (or end of epoch train)
        if not is_train:
            mlflow.log_metrics(  # type: ignore
                {f"{log_prefix}/{k}": v for k, v in metrics.items()},
                step=self.global_step,
            )

        return metrics

    def fit(self, run_id: str, epochs: int = 20):
        print(f"Starting Training on {self.device}...")
        with mlflow.start_run(run_name=run_id):  # type: ignore
            mlflow.log_params(  # type: ignore
                {
                    "batch_size": self.batch_size,
                    "context_size": self.context_size,
                    "kl_weight": self.kl_weight,
                    "optimizer": type(self.optimizer).__name__,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                train_m = self._run_epoch(self.split.train_indices, True, "1-Train")
                val_m = self._run_epoch(self.split.val_indices, False, "2-Val")
                # test_m = self._run_epoch(self.split.test_indices, False, "3-Test")

                print(
                    f"Train | Loss: {train_m['loss']:.3f} | Subj Acc: {train_m['task_acc']:.3f} | Act Acc: {train_m['task_f1']:.3f} | KL Loss: {train_m['loss_kl']:.3f}"
                )
                print(
                    f"Val   | Loss: {val_m['loss']:.3f} | Subj Acc: {val_m['task_acc']:.3f} | Act Acc: {val_m['task_f1']:.3f} | KL Loss: {val_m['loss_kl']:.3f}"
                )
                # print(
                #     f"Test  | Loss: {test_m['loss']:.3f} | Subj Acc: {test_m['task_acc']:.3f}"
                # )

        return self.model.state_dict()
