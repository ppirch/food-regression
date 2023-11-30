import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models


class FoodNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(self.n_features, self.n_features)),
                    ("relu1", nn.ReLU()),
                    ("final", nn.Linear(self.n_features, 1)),
                ]
            )
        )

    def forward(self, x):
        x = self.net(x)
        reg_head = self.net.fc1(x)
        return reg_head
