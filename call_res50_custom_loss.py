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
        self.sigma1 = nn.Parameter(torch.zeros(1))
        self.sigma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.net(x)
        reg_head = self.net.fc1(x)
        class_head = self.net.fc2(x)
        return reg_head, class_head, self.sigma1, self.sigma2

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

                model = FoodNetV1(len(le.classes_))

                model = model.to(device)

                regLoss = nn.L1Loss()
                classiLoss = nn.CrossEntropyLoss()
                
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

                model.train()
                running_loss = 0.0
                for epoch in tqdm(range(0, 40)):  # loop over the dataset multiple times
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

                        reg_head, classi_head, log_var_a, log_var_b = outputs
                        loss0 = regLoss(reg_head, weight.unsqueeze(1))
                        loss1 = classiLoss(classi_head, labels)

                        precision0 = torch.exp(-log_var_a)
                        loss0 = precision0 * loss0 + log_var_a

                        precision1 = torch.exp(-log_var_b)
                        loss1 = precision1 * loss1 + log_var_b

                        loss = loss0 + loss1
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
                        f"./model/foodnet_resnet50_custom_loss_{epoch+1}_{SEED}_.pth",
                    )
                    print("Finished Saving\n")
