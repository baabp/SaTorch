from src.exec.datasets import eurosat
from src.datasets.eurosat import EuroSAT, random_split
from src.datasets.utils import calc_normalization
from torchvision import models, transforms
from torch.utils.data import DataLoader

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchsummary import summary

from tqdm import tqdm, trange
import shutil


def train_epoch(train_dl, model, loss, optimizer, epoch, device, writer):
    model.train()
    train_dl = tqdm(train_dl, "Train", unit="batch")
    for i, (images, labels) in enumerate(train_dl):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(images)
        _loss = loss(preds, labels)
        acc = (labels == preds.argmax(1)).float().mean()

        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        writer.add_scalar('loss/train', _loss, epoch * len(train_dl) + i)
        writer.add_scalar('acc/train', acc, epoch * len(train_dl) + i)


from collections import namedtuple

TestResult = namedtuple('TestResult', 'truth predictions')


def predict(model: nn.Module, dl: torch.utils.data.DataLoader, paths=None, show_progress=True):
    """
    Run the model on the specified data.
    Automatically moves the samples to the same device as the model.
    """
    if show_progress:
        dl = tqdm(dl, "Predict", unit="batch")
    device = next(model.parameters()).device

    model.eval()
    preds = []
    truth = []
    i = 0
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        p = model(images).argmax(1).tolist()
        preds += p
        truth += labels.tolist()

        if paths:
            for pred in p:
                print(f"{paths[i]!r}, {pred}")
                i += 1

    return TestResult(truth=torch.as_tensor(truth), predictions=torch.as_tensor(preds))


def prepare_pt_context(num_gpus,
                       batch_size):
    """
    Correct batch size.
    Parameters
    ----------
    num_gpus : int
        Number of GPU.
    batch_size : int
        Batch size for each GPU.
    Returns
    -------
    bool
        Whether to use CUDA.
    int
        Batch size for all GPUs.
    """
    use_cuda = (num_gpus > 0)
    batch_size *= max(1, num_gpus)
    return use_cuda, batch_size



def main(lr=0.0001, wd=0, ratio=0.9, batch_size=32, workers=4, epochs=15):
    # Prepare dataset
    dataset = eurosat()

    # Divide to subset
    # ratio = 0.9
    trainval, test_ds = random_split(dataset, ratio, random_state=42)
    train_ds, val_ds = random_split(trainval, ratio, random_state=7)

    # Prepare DataLoader
    # batch_size = 32
    # workers = 4
    #
    # epochs = 15

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

    # images, labels = next(iter(train_dl))
    # print(images.shape, labels.shape)

    # "Calculate the mean and std of each channel on images from `train_dl`"
    mean, std = calc_normalization(train_dl)
    print(mean, std)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))

    normalization = {}
    normalization = {'mean': mean, 'std': std}

    # model:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = True

    # create/load model, changing the head for our number of classes
    model = models.resnet50(pretrained=pretrained)
    if pretrained:
        for param in model.parameters():
            # don't calculate gradient
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    model = model.to(device)
    loss = nn.CrossEntropyLoss()

    params = model.fc.parameters() if pretrained else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    print(model)
    print(summary(model, input_size=(3, 128, 128)))


    # tensorboard
    writer = SummaryWriter(log_dir="./logs")
    # display some examples in tensorboard
    images, labels = next(iter(train_dl))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    writer.add_images('images/original', originals, 0)
    writer.add_images('images/normalized', images, 0)
    writer.add_graph(model, images.to(device))

    best_acc = 0
    for epoch in trange(epochs, desc="Epochs"):
        train_epoch(train_dl, model, loss, optimizer, epoch, device, writer)
        truth, preds = predict(model, val_dl)

        torch.save(
            {'normalization': normalization, 'model_state': model.state_dict()},
            './weights/checkpoint.pt',
        )

        val_acc = (truth == preds).float().mean()
        writer.add_scalar('acc/val', val_acc, epoch * len(train_dl))
        if val_acc > best_acc:
            print(f"New best validation accuracy: {val_acc}")
            best_acc = val_acc
            shutil.copy('weights/checkpoint.pt', 'weights/best.pt')

    writer.close()


if __name__ == '__main__':

    main()
