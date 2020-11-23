import torch
import numpy as np
from scipy.fftpack import dct, idct
import os
from os import sys, path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import make_grid
from MMD_loss import MMD_loss


def DCT(image):
    return dct(dct(image, norm="ortho", axis=0), norm="ortho", axis=1)


def iDCT(image):
    return idct(idct(image, norm="ortho", axis=0), norm="ortho", axis=1)


def lscale01(M):
    New_M = np.zeros((M.shape))
    MIN = np.min(M)
    MAX = np.max(M)
    if (MAX == MIN):
        New_M[:, :] = 0.0 * M
    else:
        New_M[:, :] = (M - MIN) / (MAX - MIN)
    return New_M


class AttackPipeline:
    def __init__(self, model, batch_size, num_folds, testset, classes, adversary, loss, out_folder, target_label=3, device=torch.device('cpu')):
        self.model = model
        self.adversary = adversary
        self.loss = loss
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.classes = classes
        self.target_label = target_label
        self.out_folder = out_folder
        if not path.isdir(out_folder):
            os.makedirs(out_folder)
        self.filetxt = path.join(out_folder, 'output.txt')
        self.device = device

    def launch_attack(self):
        # Storage variables to hold on to results across batches
        true_labels = np.zeros((0)).astype(int)
        pred_cln = np.zeros((0)).astype(int)
        pred_untargeted_adv = np.zeros((0)).astype(int)
        pred_targeted_adv = np.zeros((0)).astype(int)

        self.DCT_targeted = 0
        self.DCT_untargeted = 0
        self.MMD_targeted = np.zeros((0))
        self.MMD_untargeted = np.zeros((0))

        fold = 0
        device = self.device
        model = self.model

        for cln_data, true_label in self.testloader:
            print("working for fold number :", fold+1, "\n")

            cln_data, true_label = cln_data.to(device), true_label.to(device)
            true_labels = np.concatenate(
                (true_labels, true_label.cpu().numpy().astype(int)), axis=None)

            _, pred_cln_ = torch.max(model(cln_data), 1)
            pred_cln = np.concatenate(
                (pred_cln, pred_cln_.cpu().numpy().astype(int)), axis=None)

            ####################################
            # Perform untargeted attack
            ####################################
            self.adversary.targeted = False

            print("starting untargeted attack")
            adv_untargeted = self.adversary.perturb(cln_data, true_label)
            print("Completed untargeted attack !")

            _, pred_untargeted_adv_ = torch.max(model(adv_untargeted), 1)
            pred_untargeted_adv = np.concatenate(
                (pred_untargeted_adv, pred_untargeted_adv_.cpu().numpy().astype(int)), axis=None)

            ####################################
            # perform targeted attack
            ####################################
            target = torch.ones_like(true_label) * self.target_label
            self.adversary.targeted = True

            print("starting targeted attack ...")
            adv_targeted = self.adversary.perturb(cln_data, target)
            print("Completed targeted attack !")

            _, pred_targeted_adv_ = torch.max(model(adv_targeted), 1)
            pred_targeted_adv = np.concatenate(
                (pred_targeted_adv, pred_targeted_adv_.cpu().numpy().astype(int)), axis=None)

            ####################################
            # perform DCT map analysis and MMD loss calculations
            ####################################

            self.compute_DCT_maps(cln_data, adv_targeted, adv_untargeted)

            fold += 1
            if fold >= self.num_folds:
                # select only good samples from the final fold for showcasing
                compare_prediction_label = (pred_cln_ == true_label)
                self.good_samples = [i for i, x in enumerate(
                    compare_prediction_label) if x]
                # Store the variables from the final fold for plotting

                # Labels from last batch
                self.pred_cln_ = pred_cln_
                self.pred_targeted_adv_ = pred_targeted_adv_
                self.pred_untargeted_adv_ = pred_untargeted_adv_
                self.true_label = true_label
                # Images from last batch
                self.adv_untargeted = adv_untargeted
                self.adv_targeted = adv_targeted
                self.cln_data = cln_data

                break

        # Normalize the DCT maps wrt the number of images processed
        self.DCT_untargeted /= len(true_labels)
        self.DCT_targeted /= len(true_labels)

        acc_adv_untargeted = 100 * \
            np.mean(true_labels == pred_untargeted_adv)
        acc_adv_targeted = 100*np.mean(true_labels == pred_targeted_adv)
        acc_cln = 100*np.mean(true_labels == pred_cln)

        print('Total Accuracy: ', "Clean: ", acc_cln,
              '\t and during attack: ', '\t',
              "Untargeted attack: ", acc_adv_untargeted,
              "\tTargeted attack: ", acc_adv_targeted,
              file=open(self.filetxt, "a"))

        correct_pred_indx = (true_labels == pred_cln)
        acc_adv_untargeted = 100 * \
            np.mean(true_labels[correct_pred_indx] ==
                    pred_untargeted_adv[correct_pred_indx])
        acc_adv_targeted = 100 * \
            np.mean(true_labels[correct_pred_indx] ==
                    pred_targeted_adv[correct_pred_indx])
        print('Effect of attack on accurately predcited image by unattacked model:\t'
              "Untargeted attack: ", acc_adv_untargeted,
              "\tTargeted attack: ", acc_adv_targeted,
              file=open(self.filetxt, "a"))

    def compute_DCT_maps(self, cln_data, adv_targeted, adv_untargeted):
        for i in range(cln_data.shape[0]):
            image = np.transpose(
                adv_untargeted[i].cpu().numpy(), (1, 2, 0))
            dct_adv_untargeted = (
                (DCT(image[:, :, 0]) + DCT(image[:, :, 1]) + DCT(image[:, :, 2]))/3.0)

            image = np.transpose(adv_targeted[i].cpu().numpy(), (1, 2, 0))
            dct_adv_targeted = (
                (DCT(image[:, :, 0]) + DCT(image[:, :, 1]) + DCT(image[:, :, 2]))/3.0)

            image = np.transpose(cln_data[i].cpu().numpy(), (1, 2, 0))
            dct_image = (
                (DCT(image[:, :, 0]) + DCT(image[:, :, 1]) + DCT(image[:, :, 2]))/3.0)

            self.DCT_untargeted += abs((dct_image -
                                        dct_adv_untargeted)/dct_image)
            self.DCT_targeted += abs((dct_image -
                                      dct_adv_targeted)/dct_image)

            self.MMD_targeted = np.append(self.MMD_targeted, self.loss(
                torch.from_numpy(dct_adv_targeted), torch.from_numpy(dct_image)).cpu().item())
            self.MMD_untargeted = np.append(self.MMD_untargeted, self.loss(
                torch.from_numpy(dct_adv_untargeted), torch.from_numpy(dct_image)).cpu().item())

    def plot_DCT_maps(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im1 = ax.imshow(lscale01(self.DCT_untargeted), cmap='YlOrRd')
        ax.title.set_text('untargeted')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax)
        # ax.set_axis_off()
        plt.savefig(path.join(self.out_folder,
                              'correct_DCT_untargeted.png'))
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im1 = ax.imshow(lscale01(self.DCT_targeted), cmap='YlOrRd')
        ax.title.set_text('targeted')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax)
        # ax.set_axis_off()
        plt.savefig(path.join(self.out_folder, 'correct_DCT_targeted.png'))
        # plt.show()

    def plot_MMD_hist(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.hist(self.MMD_untargeted, bins=100, weights=100 *
                 np.ones(len(self.MMD_untargeted)) / len(self.MMD_untargeted))
        plt.xlabel("MMD_untargeted")
        plt.ylabel("% of images")
        plt.title('Histogram of MMD_untargeted')
        plt.axvline(np.mean(self.MMD_untargeted), color='k',
                    linestyle='dashed', linewidth=1)
        plt.savefig(path.join(self.out_folder, 'hist_untargeted.png'))
        # plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.hist(self.MMD_targeted, bins=100, weights=100 *
                 np.ones(len(self.MMD_targeted)) / len(self.MMD_targeted))
        plt.xlabel("MMD_targeted")
        plt.ylabel("% of images")
        plt.title('Histogram of MMD_targeted')
        plt.axvline(np.mean(self.MMD_untargeted), color='k',
                    linestyle='dashed', linewidth=1)
        plt.savefig(path.join(self.out_folder, 'hist_targeted.png'))
        # plt.show()

    @staticmethod
    def _imshow(img):
        img = img / 2 + 0.5     # unnormalize
        plt.imshow(make_grid(
            img, normalize=True, scale_each=True).numpy().transpose((1, 2, 0)))

    def plot_image_comparison(self, num_plots=8):
        cln_data = self.cln_data.cpu()
        true_label = self.true_label.cpu()
        adv_untargeted = self.adv_untargeted.cpu()
        adv_targeted = self.adv_targeted.cpu()

        plt.figure(figsize=(10, 8))
        jj = 0
        img_count = 0
        while jj < num_plots:
            ii = self.good_samples[jj]
            jj += 1
            # Don't plot images that match the lable for the targetted attack
            if true_label[ii] == self.target_label:
                continue
            img_count += 1

            plt.subplot(3, num_plots, img_count)
            AttackPipeline._imshow(cln_data[ii])
            plt.title(f"clean \n pred: {self.classes[self.pred_cln_[ii]]}")
            plt.subplot(3, num_plots, img_count + num_plots)
            AttackPipeline._imshow(adv_untargeted[ii])
            plt.title(
                f"untargeted \n adv \n pred: {self.classes[self.pred_untargeted_adv_[ii]]}")
            plt.subplot(3, num_plots, img_count + num_plots * 2)
            AttackPipeline._imshow(adv_targeted[ii])
            plt.title(
                f"target {self.classes[self.target_label]} \n adv \n pred: {self.classes[self.pred_targeted_adv_[ii]]}")

        plt.tight_layout()
        plt.savefig(path.join(self.out_folder, 'test.png'))
