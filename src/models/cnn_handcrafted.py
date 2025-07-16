import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils import get_device
from src.models.base_model import BaseModel

class HandcraftedModel(nn.Module):
    def __init__(self, base_model, handcrafted_classes: int):
        super().__init__()

        self.threshold = [0.5] * base_model.out_classes

        self.base_model = base_model
        self.hc_spread = nn.Linear(handcrafted_classes, 32)
        self.out_classes = base_model.out_classes
        self.hc_fc = nn.Linear(base_model.features_num + 32, 128)
        self.fc = nn.Linear(128, base_model.out_classes)
        self.device = get_device()

    def forward(self, x, handcrafted=None):
        x = self.base_model.extract_features(x)
        if handcrafted is None:
            raise ValueError("Handcrafted features are required for this model")

        if x.dim() != 2:
            x = torch.flatten(x, start_dim=1)
            if x.size(1) != self.out_classes:
                raise ValueError(f"Output from base model expected to have size {self.out_classes}, but got {x.size(1)}")

        handcrafted = F.relu(self.hc_spread(handcrafted))
        
        comb = torch.cat([x, handcrafted], dim=1)          # (batch_size, out_classes + 32)
        
        x = F.relu(self.hc_fc(comb))                       # (batch_size, 128)
        x = self.fc(x)                                     # (batch_size, out_classes)
        return x

if __name__ == "__main__":
    print("I am the Model with handcrafted features :)")
