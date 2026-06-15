"""Simple PyTorch trainer for single-image DIGIT fabric classification."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional, Union

import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


class Trainer:
    """Small training orchestration class with wandb logging and best checkpointing."""

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.seed = int(cfg.project.seed)
        self.device = self._select_device(str(self.trainer_cfg.device))
        self.class_names = self._resolve_class_names()
        self.checkpoint_dir = Path(str(self.trainer_cfg.checkpoint_dir))
        self.best_checkpoint_path = self.checkpoint_dir / str(self.trainer_cfg.checkpoint_filename)
        self.wandb_run = None

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set deterministic seeds for Python, NumPy when present, and torch."""
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    def fit(self) -> dict[str, Optional[Union[float, int, str]]]:
        self.set_seed(self.seed)
        self._validate_setup()
        self.model.to(self.device)
        LOGGER.info(
            "Trainer setup: device=%s train=%d val=%d test=%s batch_size=%s num_workers=%s",
            self.device,
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset) if self.test_dataset is not None else "none",
            self.trainer_cfg.batch_size,
            self.trainer_cfg.num_workers,
        )

        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        val_loader = self._make_loader(self.val_dataset, shuffle=False)
        test_loader = self._make_loader(self.test_dataset, shuffle=False) if self.test_dataset is not None else None

        criterion = nn.CrossEntropyLoss()
        optimizer = self._make_optimizer()
        max_epochs = int(self.trainer_cfg.max_epochs)
        scheduler = self._make_scheduler(optimizer, total_steps=len(train_loader) * max_epochs)
        best_val_accuracy = -1.0
        best_epoch = 0

        self._init_wandb()
        try:
            LOGGER.info("Training for %d epoch(s)", max_epochs)
            for epoch in range(1, max_epochs + 1):
                train_metrics = self.train_one_epoch(train_loader, criterion, optimizer, scheduler)
                val_metrics = self.evaluate(val_loader, criterion)
                learning_rate = self._get_learning_rate(optimizer)

                metrics = {
                    "epoch": epoch,
                    "lr": learning_rate,
                    "train/lr": learning_rate,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                }

                is_best_epoch = False
                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = float(val_metrics["accuracy"])
                    best_epoch = epoch
                    is_best_epoch = True
                    self._save_checkpoint(epoch, optimizer, scheduler, best_val_accuracy)

                metrics["best/val_accuracy"] = best_val_accuracy
                metrics["best/epoch"] = best_epoch
                self._log_epoch(
                    metrics,
                    val_metrics["targets"],
                    val_metrics["predictions"],
                    step=epoch,
                    is_best=is_best_epoch,
                )
                LOGGER.info(
                    "epoch %03d/%03d lr=%.6g train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch,
                    max_epochs,
                    learning_rate,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                )

            test_accuracy = None
            if test_loader is not None and bool(getattr(self.trainer_cfg, "evaluate_test", True)):
                test_metrics = self.evaluate(test_loader, criterion)
                test_accuracy = float(test_metrics["accuracy"])
                self._wandb_log({"test/loss": test_metrics["loss"], "test/accuracy": test_accuracy})
                LOGGER.info("test_loss=%.4f test_acc=%.4f", test_metrics["loss"], test_accuracy)

            return {
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "best_checkpoint": str(self.best_checkpoint_path),
                "test_accuracy": test_accuracy,
            }
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def train_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch_size

        return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, criterion: nn.Module) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        targets: list[int] = []
        predictions: list[int] = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((preds == labels).sum().item())
            total += batch_size
            targets.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "targets": targets,
            "predictions": predictions,
        }

    def _make_loader(self, dataset: Any, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        num_workers = int(self.trainer_cfg.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(self.trainer_cfg.batch_size),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=self._seed_worker,
            generator=generator,
            persistent_workers=num_workers > 0,
        )

    def _seed_worker(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except ImportError:
            pass
        torch.manual_seed(worker_seed)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = str(getattr(self.trainer_cfg, "optimizer", "adamw")).lower()
        learning_rate = float(self.trainer_cfg.learning_rate)
        weight_decay = float(getattr(self.trainer_cfg, "weight_decay", 0.0))

        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adamw' or 'adam'.")

    def _make_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[Any]:
        scheduler_name = str(getattr(self.trainer_cfg, "lr_scheduler", "cosine")).lower()
        if scheduler_name in {"none", "off", "disabled"}:
            return None
        if scheduler_name != "cosine":
            raise ValueError(f"Unsupported lr_scheduler '{scheduler_name}'. Use 'cosine' or 'none'.")

        if total_steps < 1:
            raise ValueError("total training steps must be at least 1.")

        warmup_fraction = float(getattr(self.trainer_cfg, "lr_warmup_fraction", 0.1))
        if not 0.0 <= warmup_fraction < 1.0:
            raise ValueError("trainer.lr_warmup_fraction must be >= 0.0 and < 1.0.")

        warmup_steps = int(total_steps * warmup_fraction)
        if warmup_fraction > 0.0 and total_steps > 1:
            warmup_steps = max(1, min(warmup_steps, total_steps - 1))

        eta_min = float(getattr(self.trainer_cfg, "lr_min", 0.0))
        if warmup_steps == 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(getattr(self.trainer_cfg, "lr_warmup_start_factor", 0.001)),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    @staticmethod
    def _get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _init_wandb(self) -> None:
        wandb_cfg = getattr(self.cfg, "wandb", None)
        if wandb_cfg is None or not bool(getattr(wandb_cfg, "enabled", True)):
            return

        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        tags = getattr(wandb_cfg, "tags", [])
        init_kwargs = {
            "project": getattr(wandb_cfg, "project", None),
            "entity": getattr(wandb_cfg, "entity", None),
            "name": getattr(wandb_cfg, "name", None),
            "group": getattr(wandb_cfg, "group", None),
            "tags": list(tags) if tags is not None else [],
            "notes": getattr(wandb_cfg, "notes", None),
            "dir": getattr(wandb_cfg, "save_dir", None),
            "mode": getattr(wandb_cfg, "mode", None),
            "config": config_dict,
        }
        self.wandb_run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})
        LOGGER.info("Initialized wandb run: project=%s mode=%s", getattr(wandb_cfg, "project", None), getattr(wandb_cfg, "mode", None))

    def _log_epoch(
        self,
        metrics: dict[str, Union[float, int]],
        targets: list[int],
        predictions: list[int],
        *,
        step: int,
        is_best: bool,
    ) -> None:
        log_data: dict[str, Any] = dict(metrics)
        if self.wandb_run is not None and targets:
            import wandb

            confusion_matrix = wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets,
                preds=predictions,
                class_names=self.class_names,
            )
            log_data["val/confusion_matrix"] = confusion_matrix
            if is_best:
                log_data["best/confusion_matrix"] = confusion_matrix
        self._wandb_log(log_data, step=step)

    def _wandb_log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self.wandb_run is not None:
            if step is None:
                self.wandb_run.log(metrics)
            else:
                self.wandb_run.log(metrics, step=step)

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        val_accuracy: float,
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "best_val_accuracy": val_accuracy,
            "class_names": self.class_names,
            "seed": self.seed,
            "model_name": getattr(self.cfg.model, "name", None),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, self.best_checkpoint_path)
        LOGGER.info("Saved new best checkpoint: path=%s epoch=%d val_accuracy=%.4f", self.best_checkpoint_path, epoch, val_accuracy)

    def _validate_setup(self) -> None:
        if int(self.trainer_cfg.max_epochs) < 1:
            raise ValueError("trainer.max_epochs must be at least 1.")
        if int(self.trainer_cfg.batch_size) < 1:
            raise ValueError("trainer.batch_size must be at least 1.")
        if int(self.trainer_cfg.num_workers) < 0:
            raise ValueError("trainer.num_workers must be non-negative.")
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty.")
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty; cannot choose the best validation checkpoint.")
        if not self.class_names:
            raise ValueError("Class names are required for checkpoint metadata and wandb confusion matrices.")

    def _resolve_class_names(self) -> list[str]:
        dataset_classes = list(getattr(self.train_dataset, "classes", []))
        if dataset_classes:
            return dataset_classes

        data_cfg = getattr(self.cfg, "data", None)
        if data_cfg is not None:
            cfg_classes = getattr(data_cfg, "classes", None)
            if cfg_classes is not None:
                return list(cfg_classes)
        return []

    @staticmethod
    def _select_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("trainer.device requested CUDA, but torch.cuda.is_available() is false.")
        return selected
