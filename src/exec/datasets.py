from src.datasets.eurosat import EuroSAT, random_split
from src.datasets.utils import calc_normalization
from torchvision import models, transforms
from torch.utils.data import DataLoader


def eurosat():
    # get instance
    dataset = EuroSAT(
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    )
    return dataset


def main():
    # Prepare dataset
    dataset = eurosat()

    # Divide to subset
    ratio = 0.9
    trainval, test_ds = random_split(dataset, ratio, random_state=42)
    train_ds, val_ds = random_split(trainval, ratio, random_state=7)

    # Prepare DataLoader
    batch_size = 32
    workers = 4

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
    )

    images, labels = next(iter(train_dl))
    print(images.shape, labels.shape)

    # "Calculate the mean and std of each channel on images from `train_dl`"
    mean, std = calc_normalization(train_dl)
    print(mean, std)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))

    dict_param = {}
    dict_param = {'mean': mean, 'std': std}

    return train_dl, val_dl, dataset, dict_param

if __name__ == '__main__':
    train_dl, val_dl, dataset, dict_param = main()
