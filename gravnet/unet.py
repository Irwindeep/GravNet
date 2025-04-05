import torch
import torch.nn as nn
from torchinfo import summary

from .functional import (
    DepthwiseSeparableConv1d,
    Conv1dBlock
)
from typing import List, Tuple

class Encoder(nn.Module):
    def __init__(self, channels: List[int], kernel_size: int) -> None:
        super(Encoder, self).__init__()

        in_channels, out_channels = channels[:-2], channels[1:-1]
        self.conv_layers = nn.ModuleList([
            Conv1dBlock(in_channel, out_channel, kernel_size)
            for in_channel, out_channel in zip(in_channels, out_channels)
        ])
        self.pool_layers = nn.ModuleList([nn.MaxPool1d(4, 4) for _ in range(len(in_channels))])
        self.bottleneck = DepthwiseSeparableConv1d(channels[-2], channels[-1], kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        input, outputs = input.unsqueeze(1), []
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            input = conv_layer(input)
            outputs.append(input)
            input = pool_layer(input)

        bottleneck = self.bottleneck(input)
        return bottleneck, outputs
    
class Decoder(nn.Module):
    def __init__(self, channels: List[int], kernel_size: int) -> None:
        super(Decoder, self).__init__()

        in_channels, out_channels = channels[:-1], channels[1:]

        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=4, stride=4)
            for in_channel, out_channel in zip(in_channels[:-1], out_channels[:-1])
        ])

        self.deconv_layers = nn.ModuleList([
            Conv1dBlock(in_channel, out_channel, kernel_size)
            for in_channel, out_channel in zip(in_channels[:-1], out_channels[:-1])
        ])
        self.final_conv = DepthwiseSeparableConv1d(in_channels[-1], out_channels[-1], kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, input: torch.Tensor, skip_conns: List[torch.Tensor]) -> torch.Tensor:
        output = input
        for upconv, deconv, skip_conn in zip(self.upconv_layers, self.deconv_layers, skip_conns[::-1]):
            output = upconv(output)
            output = torch.cat([skip_conn, output], dim=1)
            output = deconv(output)

        output = self.final_conv(output)
        return output.squeeze(1)

class UNet(nn.Module):
    def __init__(self, encoder_channels: List[int], kernel_size: int) -> None:
        super(UNet, self).__init__()
        self.encoder = Encoder(encoder_channels, kernel_size)
        self.decoder = Decoder(encoder_channels[::-1], kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, skip_conns = self.encoder(input)
        output = self.decoder(output, skip_conns)

        return output
    
    def summary(self, input_size: Tuple[int, ...], depth: int = 2) -> str:
        return str(summary(self, input_size, depth=depth))

class UNetFineTuned(nn.Module):
    def __init__(self, backbone_path: str, device="cpu") -> None:
        super(UNetFineTuned, self).__init__()
        backbone = UNet([1, 32, 64, 128, 256, 512], kernel_size=3)
        backbone.load_state_dict(torch.load(backbone_path, weights_only=True, map_location=torch.device(device)))

        self.encoder = backbone.encoder
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096, 3)

    def forward(self, input):
        bottleneck, _ = self.encoder(input)
        flattened_bottleneck = self.flatten(self.pool(bottleneck))
        output = self.fc(flattened_bottleneck)
        return output
    
    def summary(self, input_size: Tuple[int, ...], depth: int = 2) -> str:
        return str(summary(self, input_size, depth=depth))
