import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from meta_learning.tracking import NullTracker, Tracker
from meta_learning.training.run_config import TrainRunConfig
from meta_learning.utils.logging import get_logger

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
        """
        Returns True if the current validation loss is an improvement.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improvement found
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        tracker: Tracker | None = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tracker = tracker or NullTracker()

        # Track global step for tracker continuity
        self.global_step = 0

    def _process_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        # Unpack: (Target, Input) - Adjust based on your dataset structure
        # Assuming standard PyTorch (x, y) or (y, x) depending on your collate_fn
        # Based on your previous snippet it looked like (y, x), but standard is (x, y).
        # We will stick to the previous snippet's order: y, x = batch
        y, x = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        output = self.model(x)
        loss = self.criterion(output, y)

        # Get predictions
        preds = output.argmax(dim=1)

        return output, loss.item(), preds, y

    def _run_epoch(
        self, loader: DataLoader, is_train: bool = True, desc: str = ""
    ) -> Dict[str, float]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        total_samples = 0

        pbar = tqdm(loader, desc=desc, leave=False)

        # No grad context for validation/test to save memory
        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for batch in pbar:
                if is_train:
                    self.optimizer.zero_grad()

                # Run batch processing
                output, loss_val, preds, targets = self._process_batch(batch)

                if is_train:
                    # Re-calculate loss tensor for backward pass
                    loss_tensor = self.criterion(output, targets)
                    loss_tensor.backward()
                    self.optimizer.step()
                    self.global_step += 1

                # --- Batch Metrics (CPU) ---
                batch_size = len(targets)
                total_loss += loss_val * batch_size
                total_samples += batch_size

                preds_cpu = preds.detach().cpu()
                targets_cpu = targets.detach().cpu()

                # Collect for epoch-level calculation
                all_preds.append(preds_cpu)
                all_targets.append(targets_cpu)

                # --- Tracker Logging (Train only: log every batch) ---
                if is_train:
                    batch_acc = (preds_cpu.numpy() == targets_cpu.numpy()).mean()
                    batch_f1 = float(
                        f1_score(
                            targets_cpu.numpy(),
                            preds_cpu.numpy(),
                            average="macro",
                            zero_division=0,
                        )
                    )

                    self.tracker.log_metrics(
                        {
                            f"{desc.lower()}/loss": loss_val,
                            f"{desc.lower()}/accuracy": batch_acc,
                            f"{desc.lower()}/f1_macro": batch_f1,
                        },
                        step=self.global_step,
                    )

                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # --- Aggregate Epoch Results ---
        avg_loss = total_loss / total_samples

        # Concatenate all batches
        all_preds_np = torch.cat(all_preds).numpy()
        all_targets_np = torch.cat(all_targets).numpy()

        # Calculate Accuracy
        accuracy = (all_preds_np == all_targets_np).mean()

        # Calculate Macro F1 Score
        f1 = float(
            f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
        )

        # --- Tracker Logging (Val/Test: log once per epoch) ---
        if not is_train:
            self.tracker.log_metrics(
                {
                    f"{desc.lower()}/loss": avg_loss,
                    f"{desc.lower()}/accuracy": accuracy,
                    f"{desc.lower()}/f1_macro": f1,
                },
                step=self.global_step,
            )

        return {"loss": avg_loss, "accuracy": accuracy, "f1_macro": f1}

    def fit(
        self,
        run_name: str,
        run_cfg: TrainRunConfig,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        epochs = run_cfg.epochs
        patience = run_cfg.patience
        early_stopper = EarlyStopping(patience=patience)
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_metrics: Dict[str, Dict[str, float]] = {}

        logger.info("Starting training on %s for %s epochs.", self.device, epochs)

        with self.tracker.start_run(run_name=run_name):
            params = run_cfg.to_tracking_dict()
            params["optimizer"] = type(self.optimizer).__name__
            params["model_class"] = type(self.model).__name__
            self.tracker.log_params(params)

            for epoch in range(epochs):
                logger.info("Epoch %s/%s", epoch + 1, epochs)

                # 1. Train
                train_metrics = self._run_epoch(
                    self.train_loader, is_train=True, desc="1-Train"
                )

                # 2. Validate
                val_metrics = self._run_epoch(
                    self.val_loader, is_train=False, desc="2-Val"
                )

                # 3. Test (Monitor performance)
                test_metrics = self._run_epoch(
                    self.test_loader, is_train=False, desc="3-Test"
                )

                # 4. Print Progress
                logger.info(
                    "Train: Loss %.4f | Acc %.4f | F1 %.4f | "
                    "Val: Loss %.4f | Acc %.4f | F1 %.4f | "
                    "Test: Loss %.4f | Acc %.4f | F1 %.4f",
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    train_metrics["f1_macro"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["f1_macro"],
                    test_metrics["loss"],
                    test_metrics["accuracy"],
                    test_metrics["f1_macro"],
                )

                # 5. Check Early Stopping & Save Best Model
                is_best = early_stopper(val_metrics["loss"])

                if is_best:
                    # Save the model state if this is the best validation loss so far
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_metrics = {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    }
                    logger.info("Best model saved.")

                if early_stopper.early_stop:
                    logger.info("Early stopping triggered after %s epochs.", epoch + 1)
                    break

        # Load the best weights back into the model before returning
        logger.info("Restoring best model weights...")
        self.model.load_state_dict(best_model_state)

        return best_model_state, best_metrics
