from torchvision import models, transforms
from torch import nn

import argparse
import os
import shutil

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchsummary import summary

def main(class_num, lr=0.0001, wd=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = True

    # create/load model, changing the head for our number of classes
    model = models.resnet50(pretrained=pretrained)
    if pretrained:
        for param in model.parameters():
            # don't calculate gradient
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, class_num)
    model = model.to(device)
    loss = nn.CrossEntropyLoss()

    params = model.fc.parameters() if pretrained else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    print(model)
    print(summary(model, input_size=(3, 128, 128)))

    return model, loss, optimizer, params

if __name__ == '__main__':
    main(class_num=4)