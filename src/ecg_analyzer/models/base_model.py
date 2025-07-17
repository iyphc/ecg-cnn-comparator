import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.utils import get_device

class BaseModel(nn.Module):
    def __init__(self, in_channels: int, out_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.features_num = 256

        self.threshold = [0.5] * out_classes

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout1d(0.25),

            nn.Conv1d(16, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout1d(0.25),

            nn.Conv1d(64, 256, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=10),
            nn.Dropout1d(0.25),

            nn.Flatten(),
            nn.Linear(256 * 10, 256),
            nn.ReLU()
        )
        
        # Добавляем финальный слой как отдельный атрибут
        self.classifier = nn.Linear(256, self.out_classes)
    def extract_features(self, x):
        x = self.seq(x)
        return x
    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    print("I am the BasicModel :)")

    
    
    