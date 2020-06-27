import os
from tqdm import tqdm, trange
import shutil
from collections import namedtuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import models, transforms
from torchsummary import summary

from src.datasets.utils import get_dataset
from src.models.utils import get_model_torchsat
from src.datasets.eurosat import EuroSAT, random_split
from src.datasets.utils import calc_normalization


def train_epoch(train_dl, model, loss, optimizer, epoch, device, writer, show_progress=True):
    """ Training network in single epoch

    Args:
        train_dl (DataLoader): DataLoader of training set
        model (nn.Module): model in PyTorch
        loss (loss): PyTorch loss
        optimizer (optimizer): PyTorch optimizer
        epoch (int): epoch number
        device (torch.device): torch.device
        writer (SummaryWriter): instance of SummaryWriter for TensorBoard
        show_progress (bool): if True, tqdm will be shown

    Returns:

    """
    model.train()
    if show_progress:
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


def evaluate_epoch(eval_dl, model, criterion, epoch, writer):
    """ evaluation in a epoch

    Args:
        eval_dl (DataLoader): DataLoader of validation set
        model (nn.Module): model in PyTorch
        criterion (loss): PyTorch loss
        epoch (int): epoch number
        writer (SummaryWriter): instance of SummaryWriter for TensorBoard

    Returns:

    """
    device = next(model.parameters()).device

    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (image, target) in enumerate(eval_dl):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(eval_dl.dataset) / eval_dl.batch_size
        val_acc = 100. * correct / len(eval_dl.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(eval_dl.dataset),
            100. * correct / len(eval_dl.dataset)))
        writer.add_scalar('loss/val', loss, len(eval_dl) * epoch)
        writer.add_scalar('acc/val', correct / len(eval_dl.dataset), epoch)

    return val_acc


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


def main(name_dataset='eurosat', lr=0.0001, wd=0, ratio=0.9, batch_size=32, workers=4, epochs=15, num_gpus=1,
         resume=None, dir_weights='./weights'):
    torch.backends.cudnn.benchmark = True
    use_cuda, batch_size = prepare_pt_context(
        num_gpus=num_gpus,
        batch_size=batch_size)

    # Prepare dataset
    # kwargs for selected dataset leading function
    kwargs = {
        'transform': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    }
    # todo: check
    #   import torchsat.transforms.transforms_cls as T_cls
    #   from torchsat.datasets.folder import ImageFolder
    dataset = get_dataset(name=name_dataset, **kwargs)

    # Divide to subset
    trainval, test_ds = random_split(dataset, ratio, random_state=42)
    train_ds, val_ds = random_split(trainval, ratio, random_state=7)

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

    # "Calculate the mean and std of each channel on images from `train_dl`"
    mean, std = calc_normalization(train_dl)
    print(mean, std)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))

    normalization = {}
    normalization = {'mean': mean, 'std': std}

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    num_classes = len(dataset.classes)
    kwargs_model = {
        'pretrained': False,
        'progress': True,
    }
    net = get_model_torchsat(name='vgg11_bn', num_classes=num_classes, **kwargs_model)

    if use_cuda:
        net = net.cuda()
    # add pretrained network
    # resume -> 'path to latest checkpoint (default: none)'
    if resume is not None:
        net.load_state_dict(torch.load(resume, map_location=device))

    # loss
    criterion = nn.CrossEntropyLoss()

    # optim and lr scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # https://qiita.com/keng000/items/c50794fb7f029062bd0d in jp
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(net)
    print(summary(net, input_size=(3, 128, 128)))

    # todo (in the future):
    #  check multi GPU use: torch.nn.DataParallel   net = torch.nn.DataParallel(net) in src/models/utils.py

    # tensorboard
    writer = SummaryWriter(log_dir="./logs")
    # display some examples in tensorboard
    images, labels = next(iter(train_dl))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    writer.add_images('images/original', originals, 0)
    writer.add_images('images/normalized', images, 0)
    writer.add_graph(net, images.to(device))

    best_acc = 0
    for epoch in trange(epochs, desc="Epochs"):
        writer.add_scalar('train/learning_rate', lr_scheduler.get_lr()[0], epoch)

        train_epoch(train_dl, net, criterion, optimizer, epoch, device, writer, show_progress=False)
        lr_scheduler.step()

        val_acc = evaluate_epoch(val_dl, net, criterion, epoch, writer)
        # truth, preds = predict(net, val_dl)
        # val_acc = (truth == preds).float().mean()
        # writer.add_scalar('acc/val', val_acc, epoch * len(train_dl))

        # save results
        path_save = os.path.join(dir_weights, 'checkpoint.pt')
        torch.save(
            {'normalization': normalization, 'model_state': net.state_dict()},
            path_save,
        )

        if val_acc > best_acc:
            print(f"New best validation accuracy: {val_acc}")
            best_acc = val_acc
            shutil.copy(path_save, os.path.join(dir_weights, 'best.pt'))

    writer.close()


if __name__ == '__main__':
    main()
