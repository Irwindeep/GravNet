import torch
import torch.nn as nn
from .functional import DenseBlock, Transition

class DenseNet(nn.Module):
    def __init__(
        self, in_channels: int, num_params: int, growth_rate: int = 32,
        block_config: tuple = (4, 4, 4, 4), compression: float = 0.5
    ) -> None:
        super().__init__()
        self.growth_rate = growth_rate
        self.compression = compression

        # Initial layers
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transitions
        num_features = 64
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition(num_features, compression)
                self.transitions.append(trans)
                num_features = int(num_features * compression)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_features, num_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
