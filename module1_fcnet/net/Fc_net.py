import torch.nn as nn
import torch

class Fc_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(3 * 50 * 50, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
            # 神经网络输入是NV结构，
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc_layer(x)