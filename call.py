from utils.datasets import FoodImagesDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from network import FoodNet

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("./regression/csv/train.csv")
    le = LabelEncoder()
    le.fit(df["food"])

    train_datasets = FoodImagesDataset(
        csv_file="./regression/csv/train.csv",
        img_dir="./regression/train",
        target_transform=le.transform,
    )
    test_datasets = FoodImagesDataset(
        csv_file="./regression/csv/test.csv",
        img_dir="./regression/test",
        target_transform=le.transform,
    )

    train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=16, shuffle=True)

    model = FoodNet(len(le.classes_))
    model = model.to(device)

    regLoss = nn.L1Loss()
    classiLoss = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    running_loss = 0.0
    for epoch in range(10):  # loop over the dataset multiple times
        print(f"Epoch: {epoch + 1}")
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
            loss1 = regLoss(reg_head, weight.unsqueeze(1))
            loss2 = classiLoss(classi_head, labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i and i % 50 == 0:  # print every 2000 mini-batches
                print(f"[{i:4d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

        print("Finished Training")
        torch.save(model.state_dict(), f"./model/foodnet_{epoch+1}.pth")
        print("Finished Saving")
