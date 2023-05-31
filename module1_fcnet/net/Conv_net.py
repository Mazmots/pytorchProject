import torch.nn as nn
import torch

class Conv_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_customize = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 512),  # 调整全连接层的输出大小为512
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model_customize(x)


