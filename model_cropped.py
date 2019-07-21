'''
# Created by Srikanth Adya at 7/16/2019

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

from sklearn.utils import shuffle

IMAGE_HEIGHT= 160
IMAGE_WIDTH = 320
IMAGE_CHANNEL = 3
BATCH_SIZE = 32
EPOCHS = 5
# fuction to read image from file
MODEL_FILE_NAME = './model_crop.h5'
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
#print(len(df),len(d_train),len(d_valid))
def random_brightness(image):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image[:,:,2] = image[:,:,2]*(0.5+np.random.uniform())
        image = np.array(image).astype('uint8')
        image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
        return image
def get_image(sample,j):
        images, angles = [],[]
        path = os.path.join('training_images',*sample[j].split('\\')[-3:])
        angle = np.float32(sample[3])
        if j == 1:
            angle+=0.2
        if j == 2:
            angle-=0.2
        try:
           image = cv2.imread(path)
           image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
           images.append(image)
           angles.append(angle)
           images.append(random_brightness(image))
           angles.append(angle)
           images.append(cv2.flip(image,1))
           angles.append(angle*-1.0)
           return images,angles
        except:
           print('imissing image ',path)
    #plt.imshow(image)
    #plt.show()

def generator(data):
    while True:
        # Randomize the indices to make an array
        #indices_arr = np.random.permutation(data.count()[0])
        # Not randomizing since the dataset is already shuffeled
        #print('len of data', len(data))
        for batch in range(0, len(data), BATCH_SIZE):
            #print(data.tail())
            # slice out the current batch according to batch-size
            current_batch = data[batch:(batch + BATCH_SIZE)]

            # initializing the arrays, x_train and y_train
            X ,y = ([],[])

            for s in current_batch:

                for camera in range(3):
                    #print(i,j)
                    images, angles = get_image(s ,camera)

                    # Appending them to existing batch
                    X.extend(images)
                    y.extend(angles)


            yield np.array(X), np.array(y)

train_gen = generator(d_train)
validation_gen = generator(d_valid)

def get_model(time_len=1):

    model = Sequential()
    model.add(Lambda(lambda x: (x/255.)-0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), data_format=None))
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.001)))
    model.add(ELU())

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))

    # Compile and train the model,
    #model.compile('adam', 'mean_squared_error')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse',metrics=['accuracy'])

    return model

if __name__ == "__main__":

      model = get_model()

      checkpoint = ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,save_best_only=True, mode='min',save_weights_only=False)
      callbacks_list = [checkpoint] #,callback_each_epoch]
      print('Training started....')

      history = model.fit_generator(train_gen,
                                    steps_per_epoch=len(d_train)//BATCH_SIZE,
                                    epochs=EPOCHS,
                                    validation_data=validation_gen,
                                    validation_steps=len(d_valid)//BATCH_SIZE,
                                    verbose=1,
                                    callbacks=callbacks_list)
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('Model accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Test'], loc='upper left')
      plt.savefig('accuracy.png')

      #  Plot training & validation loss values
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('Model loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Test'], loc='upper left')
      plt.savefig('loss.png')


