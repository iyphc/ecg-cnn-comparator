import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample(in_channels, out_channels, stride):
    seq = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm1d(out_channels),
    )
    return seq


class BasicModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, kernel_size=11):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size

        # Для stride=1 используем 'same', для stride>1 рассчитываем правильный padding
        if stride == 1:
            padding1 = "same"
        else:
            padding1 = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample_layer = None
        if self.in_channels != self.out_channels or self.stride != 1:
            self.downsample_layer = downsample(
                self.in_channels, self.out_channels, self.stride
            )

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample_layer is not None:
            identity = self.downsample_layer(identity)

        # Если размеры не совпадают, используем интерполяцию
        if x.shape != identity.shape:
            identity = F.interpolate(identity, size=x.shape[2], mode="nearest")

        return F.relu(x + identity)


class ResNet18(nn.Module):
    def __init__(
        self, in_channels: int = 12, out_classes: int = 5, kernel_size: int = 11
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.features_num = 512

        # Входной слой
        self.conv1 = nn.Conv1d(
            in_channels,
            64,
            kernel_size=kernel_size * 3,
            stride=2,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual блоки
        self.layer1 = self._make_layer(64, 64, 2, stride=1, kernel_size=kernel_size)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, kernel_size=kernel_size)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, kernel_size=kernel_size)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, kernel_size=kernel_size)

        # Классификация
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, out_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, kernel_size=11):
        layers = []

        layers.append(BasicModule(in_channels, out_channels, stride, kernel_size))

        for _ in range(1, blocks):
            layers.append(BasicModule(out_channels, out_channels, 1, kernel_size))

        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        return x
