import pickle
from glob import glob
import os
from os import path
import pdb

for folder in glob('small_batch/*e100*'):
    fname = path.join(folder, 'attack_pipeline.pkl')
    print(folder)
    if path.isfile(fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        ap = data['ap']
        from attack_pipeline import AttackPipeline
        ap.out_folder = folder
        ap.plot_DCT_maps()
