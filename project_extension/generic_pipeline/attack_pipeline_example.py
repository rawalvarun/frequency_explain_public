import torchvision
from torchvision import models, transforms
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from MMD_loss import MMD_loss
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, LBFGSAttack

from attack_pipeline import AttackPipeline


model_filename = 'mobilenet_tl/modelfile.pth'
out_folder = 'mobilenet_PGD'

batch_size = 200
num_folds = 10

loss = MMD_loss()
model = torch.load(model_filename)
# important before starting inferencing
model.eval()


transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.STL10(
    root='../data', split='test', download=True, transform=transform)

classes = ('airplane', 'bird', 'car', 'cat', 'deer',
           'dog', 'horse', 'monkey', 'ship', 'truck')

adversary_PGD = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

ap = AttackPipeline(model, batch_size, num_folds, testset,
                    classes, adversary_PGD, loss, out_folder)

ap.launch_attack()

ap.plot_DCT_maps()
ap.plot_MMD_hist()
ap.plot_image_comparison()
