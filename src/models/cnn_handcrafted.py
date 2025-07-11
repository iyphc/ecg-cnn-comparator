import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import get_device
from src.models.base_model import BaseModel

class HandcraftedModel(BaseModel):
    def __init__(self, in_channels: int, out_classes: int, handcrafted_classes: int):
        super().__init__(in_channels=in_channels, out_classes=out_classes)
        self.hc_spread = nn.Linear(handcrafted_classes, 32)
        self.hc_fc = nn.Linear(out_classes + 32, 128)
        self.fc = nn.Linear(128, out_classes)

    def forward(self, x, handcrafted=None):
        """
        x: Tensor of shape (batch_size, in_channels, seq_len) or already processed to (batch_size, out_classes)
        handcrafted: Tensor of shape (batch_size, handcrafted_classes)
        """
        x = super().forward(x)
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
