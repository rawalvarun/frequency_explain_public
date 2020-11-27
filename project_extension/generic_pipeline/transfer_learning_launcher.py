import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model_adapt import model_adapt
import os
from os import path
from transfer_learning_pipeline import TransferLearningPipeline
import argparse


def tl_launcher(model, out_folder, trainset, valset, batch_size=50, num_epochs=20, lr=0.001, momentum=0.7, classes=('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')):

    optimizer = optim.SGD(model.adapt_layers.parameters(),
                          lr=lr, momentum=momentum)

    loss_criterion = nn.CrossEntropyLoss()
    tlp = TransferLearningPipeline(model, optimizer, loss_criterion,
                                   trainset, valset, classes, batch_size, num_epochs, out_folder)

    tlp.launch_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='', type=str,
                        help='output directory in which the finetuned modelfile.pth should be stored')
    parser.add_argument('--resume_train_from', default='', type=str,
                        help='modelfile.pth from which to load a partially trained model for resuming training')
    parser.add_argument('--model_type', default='', type=str,
                        help='one of the pytorch constructor functions to download pretrained models')
    parser.add_argument('--batch_size', default=25, type=int,
                        help='batch size for minibatch SGD')
    parser.add_argument('--num_epochs', default=20,
                        type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.7, type=float, help='momentum')
    parser.add_argument('--dataset', default='STL10', type=str,
                        help='one of the dataset class constructors from pytorch')
    parser.add_argument('--classes', default=[], type=list,
                        help='list of class labels in the dataset')

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.__dict__[args.dataset](
        root='../data', split='train', download=True, transform=transform)

    valset = datasets.__dict__[args.dataset](
        root='../data', split='test', download=True, transform=transform)

    classes = tuple(trainset.classes)

    model = None
    if not args.resume_train_from:
        model = torchvision.models.__dict__[args.model_type](pretrained=True)

        model_adapt(model, args.model_type, num_classes=len(
            classes), requires_grad=True)
    else:
        model = torch.load(args.resume_train_from)

    tl_launcher(model, args.outdir, trainset, valset, args.batch_size, args.num_epochs,
                args.lr, args.momentum, classes)
