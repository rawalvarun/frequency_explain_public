import torch
import torchvision.models as models

import time
import copy 

import torchvision 
import torch.nn as nn 
import torch 
import torch.nn.functional as F 
from torchvision import transforms,models,datasets 
import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np 
from torch import optim 

import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)

model = alexnet


for params in model.parameters(): 
    params.requires_grad = False 

from collections import OrderedDict 
 

# (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )



use_gpu = torch.cuda.is_available()

batch_size = 10
transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

trainset = torchvision.datasets.STL10(root='../data',download=True, transform=transform)
dataloaders = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False, num_workers=0)


train_size = len(trainset)
classes =  ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')


# for STL10 dataset, num_classes = 10
final_num_classes = len(classes)

classifier = nn.Sequential(OrderedDict([ 
    ('dp1',nn.Dropout(0.5)), 
    ('fc1',nn.Linear(9216,4096)), 
    ('relu1',nn.ReLU()),
    ('dp2',nn.Dropout(0.5)), 
    ('fc2',nn.Linear(4096,4096)), 
    ('relu2',nn.ReLU()), 
    ('fc3',nn.Linear(4096,final_num_classes)), 
])) 
 
model.classifier = classifier 



def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloaders:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                #import pdb; pdb.set_trace()

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_size

            if torch.__version__ >= '1.6.0':
                epoch_acc = torch.true_divide(running_corrects, train_size) * 100.0
            else:
                epoch_acc = running_corrects.to(dtype=torch.float)  / float(train_size) * 100.0

            #import pdb; pdb.set_trace()

            # if phase == 'train':
            #     scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduer = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer_ft, num_epochs=10)

torch.save(model, "saved_alexnet_transfer_learnt_10classes.pth")