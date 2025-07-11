import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import get_device


class BaseModel(nn.Module):
    def __init__(self, in_channels: int, out_classes: int, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.device = get_device()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=12, stride=4)
        self.conv2 = nn.Conv1d(16, 64, kernel_size=12, stride=4)
        self.conv3 = nn.Conv1d(64, 256, kernel_size=12, stride=2)
        self.fc1 = nn.Linear(256*25, 256)
        self.fc2 = nn.Linear(256, out_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))      # (int_channels, 1000) -> (16, 248)
        x = F.relu(self.conv2(x))      # (16, 248) -> (64, 60)
        x = F.relu(self.conv3(x))      # (64, 60) -> (256, 25)
        x = torch.flatten(x, 1)        # (256, 25) -> 6400
        x = F.relu(self.fc1(x))        # 6400 → 256
        x = self.fc2(x)                # 256 → out_classes
        return x


if __name__ == "__main__":
    print("I am the BasicModel :)")

    
    
    