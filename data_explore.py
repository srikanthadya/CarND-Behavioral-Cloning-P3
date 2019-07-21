'''
# Created by Srikanth Adya at 7/21/2019

Feature: Cofiguration variables set here
# The HDF5 raw data variables are set in this file

'''
import matplotlib.pyplot as plt
#from workspace_utils import active_session
import sys
import os
import numpy as np
#import pandas as pd
import random
import cv2
import csv
import glob
from sklearn import model_selection

from keras import backend as K
from keras import models, optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

ifile_1 = open('training_images/set_1/driving_log.csv')
ifile_2 = open('training_images/set_2/driving_log.csv')
ifile_3 = open('training_images/set_3/driving_log.csv')
df_front = []
[df_front.append(line) for line in  csv.reader(ifile_1)]
[df_front.append(line) for line in  csv.reader(ifile_2)]
#[df_front.append(line) for line in  csv.reader(ifile_3)]


df = df_front
# Split data into random training and validation sets
d_train, d_valid = model_selection.train_test_split(df, test_size=.2,shuffle=True)

angle = []
image = []
for line in d_train:
    angle.append(float(line[3]))
    image.append(line[0])
plt.hist(np.array(angle),bins=100,edgecolor='black', linewidth=1.2)
plt.xticks(rotation=90)
plt.show()

fig,axs = plt.subplots(6,4,figsize=(15,10))

for i in range(6):
    for j in range(4):
        axs[i][j].imshow(cv2.imread(image[np.random.randint(0,2000)]))

plt.show()
