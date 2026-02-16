from torch import nn
from torchvision import models

from eurosat_landuse.models.simple_cnn import SimpleCNN


def create_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model name: {name}")
