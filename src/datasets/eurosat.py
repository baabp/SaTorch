import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'


def random_split(dataset, ratio=0.9, random_state=None):
    """ split dataset

    Args:
        dataset (Dataset): Dataset to be split
        ratio (float): Split rate of training and testing.
        random_state (int): The desired seed.

    Returns:

    """
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def download(root):
        file_name = os.path.basename(URL)
        # check existence of zipped file
        if not check_integrity(os.path.join(root, file_name)):
            # if not exist, download to under root
            download_and_extract_archive(URL, root, md5=MD5)

# Apparently torchvision doesn't have any loader for this so I made one
# Advantage compared to without loader: get "for free" transforms, DataLoader
# (workers), etc
class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1

def main():
    datadir = 'data/2750'

    dataset = EuroSAT()

    print('test')


if __name__ == "__main__":
    main()