import opendatasets as od
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
from PIL import Image



class CheeseDataset(Dataset):
    def __init__(self, path="fromages-images"):
        # define transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        # Load cheeses paths into self.paths
        self.folder = torchvision.datasets.ImageFolder(path, transform=self.transform)

    def __getitem__(self, idx):
        try:
            return self.folder[idx]
        except:
            first = self.folder[0]
            return torch.rand(first[0].shape)

    def __len__(self):
        return len(self.folder)


class CheeseGenerator(pl.LightningModule):
    def __init__(self):
        pass


class CheeseDiscriminator(pl.LightningModule):
    def __init__(self):
        pass


class CheeseGAN(pl.LightningModule):
    def __init__(self):
        pass


# Download data
data = od.download("https://www.kaggle.com/datasets/mathurinache/fromages-images")
