"""
Run this script to prepare the minced dataset.

This script uses the 21 classes of 100 images each.
"""


#This code is needed to import modules and files
#from the parent directory
import sys
import os

currrentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currrentdir)
sys.path.append(parentdir)
sys.path

from tqdm import tqdm as tqdm
import numpy as np
import shutil

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir


# Clean up folders
rmdir(DATA_PATH + '/merced/images_background')
rmdir(DATA_PATH + '/merced/images_evaluation')
mkdir(DATA_PATH + '/merced/images_background')
mkdir(DATA_PATH + '/merced/images_evaluation')

print('Folders created')

# Find class identities
classes = []
for root, dirnames, files in os.walk(DATA_PATH + '/UCMerced_LandUse/Images/'):
    for d in dirnames:
        classes.append(d)

classes = list(set(classes))
print(classes)

# Train/test split
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:18], classes[18:]

# Create class folders
print('Creating class folders ...')

for c in background_classes:
    mkdir(DATA_PATH + f'/merced/images_background/{c}/')
print('background folder created')

for c in evaluation_classes:
    mkdir(DATA_PATH + f'/merced/images_evaluation/{c}/')
print('evaluation folder created')

# Move images to correct location
for root, dirs, files in os.walk(DATA_PATH + '/UCMerced_LandUse/Images'):
    if len(dirs) == 0:
        class_name = root[48:]
        subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
        for f in files:
            src = f'{root}/{f}'
            dst = DATA_PATH + f'/merced/{subset_folder}/{class_name}/{f}'
            shutil.copy(src, dst)
