"""
Run this script to prepare the miniImageNet dataset.

This script uses the 100 classes of 600 images each used in the Matching Networks paper. The exact images used are
given in data/mini_imagenet.txt which is downloaded from the link provided in the paper (https://goo.gl/e3orz6).

1. Download files from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view and place in
    data/miniImageNet/images
2. Run the script
"""
#This code is needed to import modules and files
#from the parent directory
import os, sys
currrentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currrentdir)
sys.path.append(parentdir)

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir

from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os

# Clean up folders
rmdir(DATA_PATH + '/miniImageNet/images_background')
rmdir(DATA_PATH + '/miniImageNet/images_evaluation')
mkdir(DATA_PATH + '/miniImageNet/images_background')
mkdir(DATA_PATH + '/miniImageNet/images_evaluation')

print('Folders created')

# Find class identities
classes = []
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images/'):
    for f in files:
        if f.endswith('.jpg'):
            classes.append(f[:-12])

classes = list(set(classes))

# Train/test split
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:80], classes[80:]

# Create class folders
print('Creating class folders ...')

for c in background_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_background/{c}/')
print('background folder created')

for c in evaluation_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_evaluation/{c}/')
print('evaluation folder created')

# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images'):
    for f in tqdm(files, total=600*100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            # Send to correct folder
            subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
            src = f'{root}/{f}'
            dst = DATA_PATH + f'/miniImageNet/{subset_folder}/{class_name}/{image_name}'
            shutil.copy(src, dst)
