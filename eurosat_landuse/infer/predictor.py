from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class PredictorConfig:
    image_size: int
    topk: int


class Predictor:
    def __init__(
        self, model: torch.nn.Module, class_names: list[str], cfg: PredictorConfig
    ) -> None:
        self.model = model.eval()
        self.class_names = class_names
        self.cfg = cfg

        self.tfms = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.inference_mode()
    def predict(self, image_path: Path) -> dict:
        image = Image.open(image_path).convert("RGB")
        x = self.tfms(image).unsqueeze(0)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_idx = torch.topk(probs, k=min(self.cfg.topk, probs.shape[0]))
        predicted_class = self.class_names[int(torch.argmax(probs).item())]

        probabilities = {name: float(probs[i].item()) for i, name in enumerate(self.class_names)}

        return {
            "predicted_class": predicted_class,
            "topk": [
                {"class": self.class_names[int(i.item())], "prob": float(p.item())}
                for p, i in zip(top_probs, top_idx)
            ],
            "probabilities": probabilities,
        }
