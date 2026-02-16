from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_curves(
    plots_dir: Path,
    epochs: Iterable[int],
    train_loss: list[float],
    val_loss: list[float],
    val_acc: list[float],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs_list = list(epochs)

    plt.figure()
    plt.plot(epochs_list, train_loss, label="train_loss")
    plt.plot(epochs_list, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curves.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_list, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "val_acc.png")
    plt.close()


def save_confusion_matrix(
    plots_dir: Path,
    cm: np.ndarray,
    class_names: list[str],
    filename: str = "confusion_matrix.png",
    normalize: bool = True,
    title: Optional[str] = None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm_to_plot = cm.astype(np.float64)
    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_to_plot = cm_to_plot / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / filename)
    plt.close()
