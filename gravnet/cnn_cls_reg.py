import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClsReg(nn.Module):
    def __init__(self, task: str = "regression") -> None:
        super(CNNClsReg, self).__init__()

        self.task = task
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=29, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=29, stride=1, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fc_1 = nn.Linear(64 * 55, 256)
        self.fc_2 = nn.Linear(256, 128)

        self.reg_head = nn.Linear(128, 3)
        self.class_head = nn.Linear(128, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.unsqueeze(1)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        if self.task == 'regression': return self.reg_head(x)
        elif self.task == 'classification': return self.class_head(x)
        raise ValueError(f"Task - {self.task} is not supported")
