import opendatasets as od
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
from PIL import Image



class CheeseDataset(Dataset):
    def __init__(self, path="fromages-images"):
        # Load cheeses paths into self.paths
        dirs = [os.path.join(path, i) for i in os.listdir(path)]
        self.paths = []
        for i in dirs:
            cheeses = os.listdir(i)
            for c in cheeses:
                path = os.path.join(i, c)
                self.paths.append(path)
        # define transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path)
        tensor = self.transform(image)
        return tensor

    def __len__(self):
        return len(self.paths)




# Download data
data = od.download("https://www.kaggle.com/datasets/mathurinache/fromages-images")
