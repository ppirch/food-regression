import os
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset


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
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        origin = image
        weight = self.img_labels.iloc[idx, 1]
        food = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            food = self.target_transform([food])[0]
        return image, weight, food, origin


if __name__ == "__main__":
    for SEED in [677]:
        torch.manual_seed(SEED)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv("./regression/csv/train.csv")

        train_datasets = FoodImagesDataset(
            csv_file="./regression/csv/train.csv",
            img_dir="./regression/train",
            target_transform=None,
        )
        
        for i in range(train_datasets.__len__()):
            
            img, weight, food, origin = train_datasets[i]
            if food == 'food18':
                # convert tensor imgage to PIl and show
                img = T.ToPILImage()(img)
                img.show()
                origin = T.ToPILImage()(origin)
                origin.show()
                break
            