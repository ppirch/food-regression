import torch
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
        self.net.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(self.n_features, self.n_features)),
                    ("relu1", nn.ReLU()),
                    ("final", nn.Linear(self.n_features, n_classes)),
                ]
            )
        )

    def forward(self, x):
        x = self.net(x)
        reg_head = self.net.fc1(x)
        class_head = self.net.fc2(x)
        return reg_head, class_head

class FoodNetV2(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(self.n_features + n_classes, self.n_features)),
                    ("relu1", nn.ReLU()),
                    ("final", nn.Linear(self.n_features, 1)),
                ]
            )
        )
        self.net.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(self.n_features, self.n_features)),
                    ("relu1", nn.ReLU()),
                    ("final", nn.Linear(self.n_features, n_classes)),
                ]
            )
        )

    def forward(self, x):
        x = self.net(x)
        class_head = self.net.fc2(x)
        reg_head = self.net.fc1(torch.cat([x, class_head], dim=1))
        return reg_head, class_head