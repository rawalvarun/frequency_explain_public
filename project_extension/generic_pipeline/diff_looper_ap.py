

import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def plot_DCT_diffs(DCT_diff, title, out_fname):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im1 = ax.imshow(DCT_diff, cmap='YlOrRd')
    ax.title.set_text(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)
    # ax.set_axis_off()
    plt.savefig(out_fname)


main_directory = os.getcwd()

script_path = os.path.join(main_directory, "../utils/diff_DCT.py")
script_path = script_path.replace(' ', '\\ ')

# script_path = "../../../../../utils/diff_DCT.py"

# print(script_path)

attacks = ['PGD', 'DDNL', 'GSA']
models = ['alexnet', 'squeezenet', 'densenet', 'mnasnet', 'mobilenet']


# model_wise accross_attacks

model_wise_folder = os.path.join(main_directory, "DCT_DIFFs_model_wise")
create_dir(model_wise_folder)

# print(model_wise_folder)

print("\n\n MODEL-WISE : ")
for model in tqdm(models):

    model_path = os.path.join(model_wise_folder, model)

    for attack1 in tqdm(attacks):
        for attack2 in tqdm(attacks):
            if attack1 == attack2:
                continue

            foldername = f"{model}_{attack1}_vs_{attack2}"
            foldername = os.path.join(model_path, foldername)

            create_dir(foldername)

#             targeted_folder = os.path.join(foldername, "targeted")
#             create_dir(targeted_folder)
#             untargeted_folder = os.path.join(foldername, "untargeted")
#             create_dir(untargeted_folder)

            first_ap = f"./small_batch/{model}_e100{attack1}/attack_pipeline.pkl"
            second_ap = f"./small_batch/{model}_e100{attack2}/attack_pipeline.pkl"

            if not os.path.isfile(first_ap):
                print(
                    f'Attack Pipeline data file {first_ap} not found. Skipping {first_ap}x{second_ap}.\n')
                continue
            if not os.path.isfile(second_ap):
                print(
                    f'Attack Pipeline data file {second_ap} not found. Skipping {first_ap}x{second_ap}.\n')
                continue

            with open(first_ap, 'rb') as f:
                data = pickle.load(f)

            ap1 = data['ap']

            with open(second_ap, 'rb') as f:
                data = pickle.load(f)

            ap2 = data['ap']

            delta_targeted = np.log10(
                ap1.DCT_targeted)-np.log10(ap2.DCT_targeted)
            delta_untargeted = np.log10(
                ap1.DCT_untargeted)-np.log10(ap2.DCT_untargeted)

            plot_DCT_diffs(delta_targeted,
                           f'{model}: {attack1}-{attack2}', os.path.join(foldername, 'diff_targeted.png'))
            plot_DCT_diffs(delta_untargeted,
                           f'{model}: {attack1}-{attack2}', os.path.join(foldername, 'diff_untargeted.png'))


attack_wise_folder = os.path.join(main_directory, "DCT_DIFFs_attack_wise")
create_dir(attack_wise_folder)


print("\n\n ATTACK-WISE : ")
for attack in tqdm(attacks):

    attack_path = os.path.join(attack_wise_folder, attack)

    for model1 in tqdm(models):
        for model2 in tqdm(models):
            if model1 == model2:
                continue

            foldername = f"{attack}_{model1}_vs_{model2}"
            foldername = os.path.join(attack_path, foldername)

            create_dir(foldername)

            # targeted_folder = os.path.join(foldername, "targeted")
            # create_dir(targeted_folder)
            # untargeted_folder = os.path.join(foldername, "untargeted")
            # create_dir(untargeted_folder)

            first_ap = f"./small_batch/{model1}_e100{attack}/attack_pipeline.pkl"
            second_ap = f"./small_batch/{model2}_e100{attack}/attack_pipeline.pkl"

            if not os.path.isfile(first_ap):
                print(
                    f'Attack Pipeline data file {first_ap} not found. Skipping {first_ap}x{second_ap}.\n')
                continue
            if not os.path.isfile(second_ap):
                print(
                    f'Attack Pipeline data file {second_ap} not found. Skipping {first_ap}x{second_ap}.\n')
                continue

            with open(first_ap, 'rb') as f:
                data = pickle.load(f)

            ap1 = data['ap']

            with open(second_ap, 'rb') as f:
                data = pickle.load(f)

            ap2 = data['ap']

            delta_targeted = np.log10(
                ap1.DCT_targeted)-np.log10(ap2.DCT_targeted)
            delta_untargeted = np.log10(
                ap1.DCT_untargeted)-np.log10(ap2.DCT_untargeted)
            plot_DCT_diffs(delta_targeted,
                           f'{attack}: {model1}-{model2}', os.path.join(foldername, 'diff_targeted.png'))
            plot_DCT_diffs(delta_untargeted,
                           f'{attack}: {model1}-{model2}', os.path.join(
                               foldername, 'diff_untargeted.png'))
