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

model = models.mnasnet1_0(pretrained=True)

model_adapt(model, 'mnasnet', num_classes=10, requires_grad=True)

optimizer = optim.SGD(model.adapt_layers.parameters(), lr=0.001, momentum=0.7)

loss_criterion = nn.CrossEntropyLoss()

batch_size = 5
num_epochs = 20
out_folder = 'mnasnet_tl'

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.STL10(
    root='../data', split='train', download=True, transform=transform)

valset = torchvision.datasets.STL10(
    root='../data', split='test', download=True, transform=transform)

classes = ('airplane', 'bird', 'car', 'cat', 'deer',
           'dog', 'horse', 'monkey', 'ship', 'truck')

tlp = TransferLearningPipeline(model, optimizer, loss_criterion,
                               trainset, valset, classes, batch_size, num_epochs, out_folder)

tlp.launch_training()
