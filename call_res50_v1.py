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


class FoodNetBaseline(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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


class FoodNetV1(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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

                train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True)

                if model_type == "baseline":
                    model = FoodNetBaseline(len(le.classes_))
                elif model_type == "V1":
                    model = FoodNetV1(len(le.classes_))
                elif model_type == "V2":
                    model = FoodNetV2(len(le.classes_))
                    
                model = model.to(device)
                regLoss = nn.L1Loss()
                classiLoss = nn.CrossEntropyLoss()

                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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

                        if model_type == "baseline":
                            reg_head = outputs
                            loss = regLoss(reg_head, weight.unsqueeze(1))
                        else:
                            reg_head, classi_head = outputs
                            loss1 = regLoss(reg_head, weight.unsqueeze(1))
                            loss2 = classiLoss(classi_head, labels)
                            loss = wr * loss1 + wc * loss2
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        if i and i % 50 == 0:  # print every 2000 mini-batches
                            print(f"[{i:4d}] loss: {running_loss / 2000:.5f}")
                            running_loss = 0.0
                    print("Finished Training")
                    torch.save(
                        model.state_dict(),
                        f"./model/foodnet_resnet50_{model_type}_{epoch+1}_{SEED}_{int(wr*100)}_{int(wc*100)}.pth",
                    )
                    print("Finished Saving\n")
