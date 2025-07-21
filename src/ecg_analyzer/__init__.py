from importlib import metadata as _md

try:
    __version__ = _md.version(__name__)
except _md.PackageNotFoundError:  # dev-режим
    __version__ = "0.1.0.dev0"

from .data.loader import get_dataloaders
from .models.base_model import BaseModel
from .models.resnet18 import ResNet
from .models.cnn_handcrafted import HandcraftedModel
from .training.trainer import train_model, validate
from .training.evaluator import evaluate_model

__all__ = [
    "__version__",
    "get_dataloaders",
    "BaseModel",
    "ResNet",
    "HandcraftedModel",
    "train_model",
    "validate",
    "evaluate_model",
]
