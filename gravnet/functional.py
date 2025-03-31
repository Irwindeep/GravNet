import torch
import torch.nn as nn

# Functions required for UNet
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super(DepthwiseSeparableConv1d, self).__init__()

        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, padding=padding
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(input))

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_blocks: int = 2) -> None:
        super(Conv1dBlock, self).__init__()

        in_channs, out_channs = [in_channels, *[out_channels]*(num_blocks-1)], [out_channels]*num_blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                DepthwiseSeparableConv1d(
                    in_channs[i], out_channs[i], kernel_size=kernel_size,
                    padding=(kernel_size-1)//2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            for i in range(num_blocks)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.blocks(input)

# Functions required for DenseNet implementation
class DenseLayer(nn.Module):
    def __init__(self, input_channels: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(input_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat([x, out], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, input_channels: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = input_channels
        for _ in range(num_layers):
            layer = DenseLayer(current_channels, growth_rate, bn_size)
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    def __init__(self, input_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(input_channels, int(input_channels * compression), kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
