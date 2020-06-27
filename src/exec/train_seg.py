import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.utils import get_dataset
import src.transforms.transforms_seg as T_seg
from src.models.utils import get_model_torchsat
from torchsummary import summary
from src.datasets.utils import calc_normalization
from ignite.metrics import IoU, Precision, Recall


def train_epoch(train_dl, model, criterion, optimizer, epoch, device, writer, show_progress=True):
    """ Training network in single epoch

    Args:
        train_dl (DataLoader): DataLoader of training set
        model (nn.Module): model in PyTorch
        criterion (loss): PyTorch loss
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
    for i, (images, targets) in enumerate(train_dl):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, i + 1, len(train_dl), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(train_dl) * epoch + i)


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
    print('\neval epoch {}'.format(epoch))
    device = next(model.parameters()).device

    model.eval()
    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(eval_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(1)

            precision.update((preds, targets))
            recall.update((preds, targets))
            mean_loss.append(loss.item())
            mean_recall.append(recall.compute().item())
            mean_precision.append(precision.compute().item())

            # print('val-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx + 1, len(dataloader), loss.item()))
            writer.add_scalar('test/loss', loss.item(), len(eval_dl) * epoch + idx)

    mean_precision, mean_recall = np.array(mean_precision).mean(), np.array(mean_recall).mean()
    f1 = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-20)

    print('precision: {:07.5}, recall: {:07.5}, f1: {:07.5}\n'.format(mean_precision, mean_recall, f1))
    writer.add_scalar('test/epoch-loss', np.array(mean_loss).mean(), epoch)
    writer.add_scalar('test/f1', f1, epoch)
    writer.add_scalar('test/precision', mean_precision, epoch)
    writer.add_scalar('test/recall', mean_recall, epoch)


def main(name_dataset='voc', batch_size=8, model="unet34", num_classes=256, pretrained=True, resume=None, lr=0.001,
         size=448, epochs=10, dir_ckp='./ckp', debug=False):
    torch.backends.cudnn.benchmark = True
    if debug:
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform
    train_transform = T_seg.Compose([
        T_seg.RandomCrop(size),  # <- should be match to U-Net or other networks' size
        T_seg.RandomHorizontalFlip(),
        T_seg.RandomVerticalFlip(),
        T_seg.ToTensor(),
        T_seg.Normalize(),
    ])
    val_transform = T_seg.Compose([
        T_seg.ToTensor(),
        T_seg.Normalize(),
    ])
    train_data = get_dataset(name=name_dataset, mode="train", transform=train_transform)
    val_data = get_dataset(name=name_dataset, mode="val", transform=val_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    print(train_data.__getitem__(1))
    # images, labels = next(iter(train_loader))

    # model
    model = get_model_torchsat(model, num_classes, pretrained=pretrained)
    model.to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(model)
    print(summary(model, input_size=(3, size, size), device=device))
    mean, std = calc_normalization(train_loader)
    normalization = {}
    normalization = {'mean': mean, 'std': std}

    writer = SummaryWriter(log_dir="./logs")
    # display some examples in tensorboard
    images, labels = next(iter(train_loader))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    writer.add_images('images/original', originals, 0)
    writer.add_images('images/normalized', images, 0)
    writer.add_graph(model, images.to(device))

    for epoch in range(epochs):
        writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)
        train_epoch(train_loader, model, criterion, optimizer, epoch, device, writer, show_progress=True)
        evaluate_epoch(val_loader, model, criterion, epoch, writer)
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(dir_ckp, 'cls_epoch_{}.pth'.format(epoch)))


if __name__ == "__main__":
    main()
