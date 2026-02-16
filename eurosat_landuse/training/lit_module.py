from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from eurosat_landuse.utils.plots import save_confusion_matrix


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    weight_decay: float


class LanduseLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        train_cfg: TrainConfig,
        class_names: list[str],
        plots_dir: Path,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_cfg = train_cfg
        self.class_names = class_names
        self.plots_dir = plots_dir

        self.acc = MulticlassAccuracy(num_classes=num_classes)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.cm = MulticlassConfusionMatrix(num_classes=num_classes)

        self._epoch_train_losses: list[float] = []
        self._epoch_val_losses: list[float] = []
        self._epoch_val_acc: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get("train/loss_epoch")
        val_loss = metrics.get("val/loss")
        val_acc = metrics.get("val/acc")

        if train_loss is not None:
            self._epoch_train_losses.append(float(train_loss.detach().cpu().item()))
        if val_loss is not None:
            self._epoch_val_losses.append(float(val_loss.detach().cpu().item()))
        if val_acc is not None:
            self._epoch_val_acc.append(float(val_acc.detach().cpu().item()))

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)

        self.cm.update(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/f1_macro", f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        cm = self.cm.compute().detach().cpu().numpy().astype(np.int64)
        save_confusion_matrix(
            plots_dir=self.plots_dir,
            cm=cm,
            class_names=self.class_names,
            filename="confusion_matrix.png",
            normalize=True,
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def epoch_curves(self) -> dict[str, list[float]]:
        return {
            "train_loss": self._epoch_train_losses,
            "val_loss": self._epoch_val_losses,
            "val_acc": self._epoch_val_acc,
        }
