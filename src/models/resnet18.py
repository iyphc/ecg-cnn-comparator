import torch
import torch.nn as nn
import torch.functional as f

class BasicModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, kernel_size=12):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        
    def downsample(self, x):
        seq = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False),
            nn.BatchNorm1d(self.out_channels)
        )
        return seq
    def forward(self, x):
        input = x.copy()
        x = self.seq(x)
        if self.in_channels != self.out_channels:
            input = downsample(input)
        return x + input

class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, kernel_size=12):
        super().__init__()
        self.seq = nn.Sequential(
            # Входной слой
            nn.Conv1d(in_channels, out_channels, kernel_size*3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=kernel_size)

            # Система блоков
            
            '''
            IN PROGRESS
            '''
        )
        