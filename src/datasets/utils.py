import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eurosat import EuroSAT

datasets = {
    'eurosat': EuroSAT
}

def get_dataset(name, **kwargs):
    print(kwargs)
    if name.lower() not in datasets:
        raise ValueError("no dataset named {}, should be one of {}".format(name, ' '.join(datasets)))
    return datasets.get(name.lower())(**kwargs)

def calc_normalization(train_dl: torch.utils.data.DataLoader):
    "Calculate the mean and std of each channel on images from `train_dl`"
    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    for images, labels in tqdm(train_dl, "Compute normalization"):
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()