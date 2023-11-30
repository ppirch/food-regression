import torch
import numpy as np
import pandas as pd
from network import FoodNetV2
from utils.datasets import FoodImagesDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_percentage_error

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
    test_dataloader = DataLoader(test_datasets, batch_size=510, shuffle=False)

    for epoch in tqdm(range(20)):  # loop over the dataset multiple times
        print(f"Epoch: {epoch + 1}")
        model = FoodNetV2(len(le.classes_))
        model.load_state_dict(torch.load( f"./model/foodnet_res50_V2_{epoch+1}_6_3.pth"))
        model.to(device)

        model.eval()
        reg = np.array([])
        classi = np.array([])
        reg_true = np.array([])
        classi_true = np.array([])

        for i, data in enumerate(test_dataloader, 0):
            inputs, weight, labels = data
            reg_true = np.concatenate((reg_true, weight.numpy()))
            classi_true = np.concatenate((classi_true, labels.numpy()))
            weight, labels = weight.long(), labels.long()
            inputs, weight, labels = inputs.to(device), weight.to(device), labels.to(device)

            outputs = model.forward(inputs)
            reg_head, classi_head = outputs

            reg_head = reg_head.squeeze_(1).detach().cpu().numpy()
            classi_head = torch.argmax(classi_head, 1)
            classi_head = classi_head.detach().cpu().numpy()

            reg = np.concatenate((reg, reg_head))
            classi = np.concatenate((classi, classi_head))

        print("MAPE: ", mean_absolute_percentage_error(reg_true, reg))
        print(classification_report(classi_true, classi, zero_division=0))
