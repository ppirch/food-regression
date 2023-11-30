from utils.datasets import FoodImagesDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import models


class FoodNetV2(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(self.n_features + n_classes, self.n_features + n_classes)),
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


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum


if __name__ == "__main__":
    weight_loss_ratio = [(0.9, 0.1)]
    for wr, wc in weight_loss_ratio:
        for model_type in ["V1"]:
            print("model: ", model_type)
            for SEED in [677]:
                torch.manual_seed(SEED)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                df = pd.read_csv("./regression/csv/train.csv")
                le = LabelEncoder()
                le.fit(df["food"])

                train_datasets = FoodImagesDataset(
                    csv_file="./regression/csv/train.csv",
                    img_dir="./regression/train",
                    target_transform=le.transform,
                )

                train_dataloader = DataLoader(
                    train_datasets, batch_size=32, shuffle=True
                )

                model = FoodNetV2(len(le.classes_))

                model = model.to(device)

                awl = AutomaticWeightedLoss(2)
                regLoss = nn.L1Loss()
                classiLoss = nn.CrossEntropyLoss()

                optimizer = optim.SGD(
                    [
                        {"params": model.parameters()},
                        {"params": awl.parameters(), "weight_decay": 0},
                    ],
                    lr=0.001,
                    momentum=0.9,
                )

                model.train()
                running_loss = 0.0
                for epoch in tqdm(range(0, 50)):  # loop over the dataset multiple times
                    print(f"\nEpoch: {epoch + 1}")
                    for i, data in enumerate(train_dataloader, 0):

                        # get the inputs; data is a list of [inputs, labels]
                        inputs, weight, labels = data
                        weight, labels = weight.long(), labels.long()
                        inputs, weight, labels = (
                            inputs.to(device),
                            weight.to(device),
                            labels.to(device),
                        )
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = model.forward(inputs)

                        reg_head, classi_head = outputs
                        loss0 = regLoss(reg_head, weight.unsqueeze(1))
                        loss1 = classiLoss(classi_head, labels)

                        loss = awl(loss0, loss1)

                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        if i and i % 50 == 0:  # print every 2000 mini-batches
                            print(f"[{i:4d}] loss: {running_loss / 2000:.5f}")
                            running_loss = 0.0
                            print(
                                f"weight1: {awl.params[0].item():.5f}, weight2: {awl.params[1].item():.5f}"
                            )

                    print("Finished Training")
                    torch.save(
                        model.state_dict(),
                        f"./model/foodnet_resnet50_custom_loss_{epoch+1}_{SEED}_v2.pth",
                    )
                    print("Finished Saving\n")
