import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from torchvision import models
from collections import OrderedDict


class FoodImagesDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: callable = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        target_transform: callable = None,
    ):
        """Food Images Dataset
        Args:
            csv_file (string): Path to the csv file with labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied
                on a sample. Defaults to None.
            target_transform (callable, optional): Transform to be applied
                on the target. Defaults to None.
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        weight = self.img_labels.iloc[idx, 1]
        food = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            food = self.target_transform([food])[0]
        return image, weight, food


class FoodNet(nn.Module):
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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("./regression/csv/train.csv")
    le = LabelEncoder()
    le.fit(df["food"])

    test_datasets = FoodImagesDataset(
        csv_file="./regression/csv/test.csv",
        img_dir="./regression/test",
        target_transform=le.transform,
    )
    test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False)
    model = FoodNet(len(le.classes_))
    model.load_state_dict(torch.load(f"./model/foodnet_resnet50_baseline.pth"))
    model.to(device)

    model.eval()
    reg = np.array([])
    reg_true = np.array([])

    for data in tqdm(test_dataloader):
        inputs, weight, labels = data
        reg_true = np.concatenate((reg_true, weight.numpy()))
        weight = weight.long()
        inputs, weight = inputs.to(device), weight.to(device)

        outputs = model.forward(inputs)
        reg_head = outputs

        reg_head = reg_head.squeeze_(1).detach().cpu().numpy()

        reg = np.concatenate((reg, reg_head))

    print("MAPE: ", mean_absolute_percentage_error(reg_true, reg))
