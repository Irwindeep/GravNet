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


