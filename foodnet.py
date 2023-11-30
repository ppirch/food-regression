import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models


def get_backbone(backbone: str = "resnet50"):
    if backbone == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif backbone == "resnet101":
        return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif backbone == "resnet152":
        return models.resnet152(weights=models.ResNet152_Weights.DEFAULT)


class ClassifyNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", n_classes: int = 5):
        super().__init__()
        self.net = get_backbone(backbone)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(
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
        class_head = self.net.fc1(x)
        return class_head


class RegressNet(nn.Module):
    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        self.net = get_backbone(backbone)
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


class SharedNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", n_classes: int = 5):
        super().__init__()
        self.net = get_backbone(backbone)
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


class ConcatNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", n_classes: int = 5):
        super().__init__()
        self.net = get_backbone(backbone)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear",
                        nn.Linear(
                            self.n_features + n_classes,
                            self.n_features + n_classes,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    ("final", nn.Linear(self.n_features + n_classes, 1)),
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
