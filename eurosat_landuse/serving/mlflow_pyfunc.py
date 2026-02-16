from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

from eurosat_landuse.models.factory import create_model


@dataclass(frozen=True)
class ServingConfig:
    data_dir: Path
    model_name: str
    num_classes: int
    image_size: int


class EuroSATPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, cfg: ServingConfig) -> None:
        self.cfg = cfg
        self._model: torch.nn.Module | None = None
        self._class_names: list[str] = []
        self._tfms = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        weights_path = Path(context.artifacts["weights"])
        ds = datasets.ImageFolder(root=str(self.cfg.data_dir))
        self._class_names = list(ds.classes)

        model = create_model(
            name=self.cfg.model_name,
            num_classes=self.cfg.num_classes,
            pretrained=False,
        )
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()

        self._model = model

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        if "image_path" not in model_input.columns:
            raise ValueError("Input DataFrame must contain column 'image_path'")

        results: list[dict[str, Any]] = []
        for image_path_str in model_input["image_path"].tolist():
            image_path = Path(str(image_path_str))
            image = Image.open(image_path).convert("RGB")
            x = self._tfms(image).unsqueeze(0)

            with torch.inference_mode():
                logits = self._model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)

            pred_idx = int(torch.argmax(probs).item())
            results.append(
                {
                    "predicted_class": self._class_names[pred_idx],
                    "probabilities": {
                        name: float(probs[i].item()) for i, name in enumerate(self._class_names)
                    },
                }
            )

        return pd.DataFrame(results)
