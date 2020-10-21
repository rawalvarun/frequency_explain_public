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

import matplotlib.pyplot as plt 



from argparse import ArgumentParser

parser = ArgumentParser(description="Generic experiment execution script")
parser.add_argument("-f", "--spec-file", type=str,
                    help="Path to experiment specification JSON file")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Make the pogram more verbose")


# Load params from the Specification JSON config file
_cnn_model_ = experiment_spec_json["CNN_Model"]
_cnn_model_id_ = experiment_spec_json["model_id"]

model = alexnet

from collections import OrderedDict 
 

use_gpu = torch.cuda.is_available()

batch_size = 10
transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

trainset = torchvision.datasets.STL10(root='../data',split='train', download=True, transform=transform)
trainloaders = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False, num_workers=0)

valset = torchvision.datasets.STL10(root='../data',split='test', download=True, transform=transform)
valloaders = torch.utils.data.DataLoader(valset, batch_size=batch_size,shuffle=False, num_workers=0)


train_size = len(trainset)
val_size = len(valset)

classes =  ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')


# for STL10 dataset, num_classes = 10
final_num_classes = len(classes)



# to control whether to just fine tune the last layers or to train the whole network
# need to adjust num_epochs accordingly
fine_tune_only = False

if fine_tune_only:
    _num_epochs_ = 10
else:
    _num_epochs_ = 20



if fine_tune_only:
    for params in model.parameters(): 
        params.requires_grad = False 
 
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

for param in model.classifier.parameters():
    param.requires_grad = True


plt.ion()
plt.show()

fig, (axs) = plt.subplots(nrows=2, figsize=(10, 5))

fig.tight_layout()

axs[0].set_autoscale_on(True)
axs[1].set_autoscale_on(True)




# naming the x axis 
# naming the y axis 
axs[0].set(ylabel='Accuracy %', xlabel='Epochs')
axs[1].set(ylabel='Loss (Cross Entropy)', xlabel='Epochs')


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    iterations_vector = []
    accuracy_vector = {}
    loss_vector = {}
    datasize = {}

    for phase in ['train', 'val']:
        accuracy_vector[phase] = []
        loss_vector[phase] = []

    datasize['train'] = train_size
    datasize['val'] = val_size

    markers = {}
    markers['train'] = '-r'
    markers['val'] = '-b'

    for epoch in range(num_epochs):

        iterations_vector.append(epoch)
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train(True)  # Set model to training mode
                dataloaders = trainloaders
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()
                dataloaders = valloaders

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

            epoch_loss = running_loss / datasize[phase]

            if torch.__version__ >= '1.6.0':
                epoch_acc = torch.true_divide(running_corrects, datasize[phase]) * 100.0
            else:
                epoch_acc = running_corrects.to(dtype=torch.float)  / float(datasize[phase]) * 100.0

            loss_vector[phase].append(epoch_loss)
            accuracy_vector[phase].append(epoch_acc)
            
            # plotting the points  
            accuracy_line, = axs[0].plot(iterations_vector, accuracy_vector[phase], markers[phase]) 
            loss_line, = axs[1].plot(iterations_vector, loss_vector[phase], markers[phase]) 
            
            accuracy_line.set_ydata(accuracy_vector[phase])
            loss_line.set_ydata(loss_vector[phase])

            axs[0].relim()
            axs[0].autoscale_view(True,True,True)
            axs[0].relim()
            axs[1].autoscale_view(True,True,True)

            plt.draw()
            plt.pause(0.001)

            # function to show the plot 
            plt.show() 

            fig.tight_layout()


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

if fine_tune_only:
    # use only classifier parameters
    optimizer_ft = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
else:
    # use all parameters
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduer = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer_ft, num_epochs=_num_epochs_)

torch.save(model, "saved_alexnet_trans_learnt_fineTune="+str(fine_tune_only)+"_.pth")

plt.savefig('transfer_learning_train.png')