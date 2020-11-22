import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import optim
import time
import copy
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import os
from os import path

import matplotlib.pyplot as plt


class TransferLearningPipeline:
    def __init__(self, model, optimizer, loss_criterion, trainset, valset, classes, batch_size, num_epochs, out_folder):
        self.model = model
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion

        self.trainset = trainset
        self.valset = valset
        self.train_size = len(trainset)
        self.val_size = len(valset)

        self.trainloaders = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.valloaders = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.classes = classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.out_folder = out_folder
        if not path.isdir(out_folder):
            os.makedirs(out_folder)

        # Plotting setup
        plt.ion()
        plt.show()

        self.fig, (self.axs) = plt.subplots(nrows=2, figsize=(10, 5))

        self.fig.tight_layout()

        self.axs[0].set_autoscale_on(True)
        self.axs[1].set_autoscale_on(True)

        # naming the x axis
        # naming the y axis
        self.axs[0].set(ylabel='Accuracy %', xlabel='Epochs')
        self.axs[1].set(ylabel='Loss (Cross Entropy)', xlabel='Epochs')

        # Plotting line markers
        self.markers = {}
        self.markers['train'] = '-r'
        self.markers['val'] = '-b'

        self.use_gpu = torch.cuda.is_available()

    def launch_training(self):
        since = time.time()

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0

        self.iterations_vector = []
        self.accuracy_vector = {}
        self.loss_vector = {}
        self.datasize = {}

        for phase in ['train', 'val']:
            self.accuracy_vector[phase] = []
            self.loss_vector[phase] = []

        self.datasize['train'] = self.train_size
        self.datasize['val'] = self.val_size

        for epoch in range(self.num_epochs):

            self.iterations_vector.append(epoch)

            print(f'Epoch {epoch+1}/{self.num_epochs}'.center(50))
            print('-' * 100)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                if phase == 'train':
                    self.model.train(True)  # Set model to training mode
                    dataloaders = self.trainloaders
                else:
                    self.model.train(False)  # Set model to evaluate mode
                    self.model.eval()
                    dataloaders = self.valloaders

                running_loss = 0.0
                running_corrects = 0.0

                # Iterate over data.
                for data in dataloaders:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)

                    # import pdb; pdb.set_trace()

                    _, preds = torch.max(outputs.data, 1)
                    loss = self.loss_criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.datasize[phase]

                if torch.__version__ >= '1.6.0':
                    epoch_acc = torch.true_divide(
                        running_corrects, self.datasize[phase]) * 100.0
                else:
                    epoch_acc = running_corrects.to(
                        dtype=torch.float) / float(self.datasize[phase]) * 100.0

                self.loss_vector[phase].append(epoch_loss)
                self.accuracy_vector[phase].append(epoch_acc)

                self.plot_train_progress(phase)

                # import pdb; pdb.set_trace()

                # if phase == 'train':
                #     scheduler.step()

                print(
                    f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'.center(50))

                # deep copy the model
                if phase == 'val' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(
                        self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.3f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(f'Best val Acc: {self.best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model, path.join(self.out_folder, 'modelfile.pth'))
        return self.model

    def plot_train_progress(self, phase, pausetime=0.001):
        # plotting the points
        accuracy_line, = self.axs[0].plot(
            self.iterations_vector, self.accuracy_vector[phase], self.markers[phase])
        loss_line, = self.axs[1].plot(
            self.iterations_vector, self.loss_vector[phase], self.markers[phase])

        accuracy_line.set_ydata(self.accuracy_vector[phase])
        loss_line.set_ydata(self.loss_vector[phase])

        self.axs[0].relim()
        self.axs[0].autoscale_view(True, True, True)
        self.axs[0].relim()
        self.axs[1].autoscale_view(True, True, True)

        plt.draw()
        plt.pause(pausetime)

        # function to show the plot
        plt.show()

        self.fig.tight_layout()
        plt.savefig(path.join(self.out_folder, 'transfer_learning_train.png'))
