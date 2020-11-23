import argparse
import os
from os import path
from attack_pipeline import AttackPipeline
from torchvision import models, transforms
import torchvision
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from MMD_loss import MMD_loss
from attack_maps import map_config_to_attack
import pickle

attacks = [
    'LinfPGDAttack',
    'CarliniWagnerL2Attack',
    'JacobianSaliencyMapAttack',
    'LBFGSAttack'
]


def attack_launcher(model_filename, out_folder, attack_name, batch_size, num_folds):

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

    adversary = map_config_to_attack(attack_name, model, classes)

    ap = AttackPipeline(model, batch_size, num_folds, testset,
                        classes, adversary, loss, out_folder)

    ap.launch_attack()

    ap.plot_DCT_maps()
    ap.plot_MMD_hist()
    ap.plot_image_comparison()

    with open(path.join(out_folder, 'attack_pipeline.pkl'), 'wb') as f:
        pickle.dump({'ap': ap}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='', type=str,
                        help=f'attack name, one of {attacks}')
    parser.add_argument('--modelfile', default='', type=str,
                        help=f'filename from where the fintuned model can be loaded')
    parser.add_argument(
        '--outdir', default='', type=str, help=f'output directory where the attack results can be stored (a new directory will be created if it does not exist)')
    parser.add_argument('--num_folds', default=1, type=int,
                        help='number of batches of the test set to generate adversarial examples for')
    parser.add_argument('--batch_size', default=200, type=int,
                        help='number of images in a testset batch')

    args = parser.parse_args()

    model_filename = args.modelfile
    if not model_filename or not path.isfile(model_filename):
        raise FileNotFoundError

    out_folder = args.outdir
    if not out_folder:
        print('Please specify --outdir')
        raise Exception

    attack_name = args.attack
    if attack_name not in attacks:
        print(f'Please specify a valid attack from one of {attacks}')
        raise Exception

    batch_size = args.batch_size
    num_folds = args.num_folds

    attack_launcher(model_filename, out_folder,
                    attack_name, batch_size, num_folds)
