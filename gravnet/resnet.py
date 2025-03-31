import torch
import torch.nn as nn
from .functional import ResidualBlock

class ResNet(nn.Module):
    def __init__(self, in_channels: int, num_params: int) -> None:
        """
        Args:
            in_channels: Number of input channels (e.g. 1 for a 1D waveform).
            num_params: Number of regression parameters to estimate.
        """
        super().__init__()
        self.conv1: nn.Conv1d = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1: nn.Sequential = self._make_layer(64, 64, blocks=2)
        self.layer2: nn.Sequential = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3: nn.Sequential = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4: nn.Sequential = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        self.fc: nn.Linear = nn.Linear(512, num_params)

    def _make_layer(self, in_planes: int, out_planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock(in_planes, out_planes, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.fc(x)
        return x
