import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import cv2

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

input_images = []

# pathnames to sample input images from project definition
# note: this dataset was modified for training; see project writeup for additional details
# header information removed before loading files manually, will cause error

def load_images(csv_filepath):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            input_images.append(line)
    return input_images

input_images = load_images('bleh/driving_log.csv')     

# separate images for validation set and use remainder for training
train_samples, validation_samples = train_test_split(input_images, test_size=0.1)

# generator function to prepare input data for keras training loop; assistance needed
# with syntax and behaviour from forum posts (referenced)

def generator(input_images, batch_size=32):
    num_samples = len(input_images)
    while 1: 
        shuffle(input_images)
        for offset in range(0, num_samples, batch_size):
            batch_samples = input_images[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                name = './bleh/'+batch_sample[0]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
              
            data = shuffle(X_train, y_train)

            yield data

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# handle image resize for preprocessing
def resize_comma(image):
    import tensorflow as tf  
    return tf.image.resize_images(image, (40, 160))


# keras model definition
model = Sequential()

# apply the resize tranformation and normalize the images (255 max, mean value), 3-channel
# crop the image headers as they are used for information display
# resize function is picky about syntax; see referenced discussions

model.add(Cropping2D(cropping=((70, 25), (0, 0)), dim_ordering='tf', input_shape=(160, 320, 3)))
model.add(Lambda(resize_comma))
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# 2D convolution layers. Per the nvidia driving model, increasing the size of the stacked 
# convolution layers allows for features to be captured. Rest of network is modelled after
# CNN structure previously used. Note: The actual shape of the network mattered much less
# than the quality of the training data! varying parameters and sizes of connected layers
# did not matter as much as expected; successful navigation required editing the training
# data. 

# data collected manually was not useful as the input was too erratic. See project discussion
# for additional commentary.

# several approaches were tried, ultimately the structures as suggested by the nvidia paper
# seemed to work best

model.add(Convolution2D(16, 8, 8, subsample=(4,4), border_mode="same"))
model.add(ELU())

model.add(Convolution2D(32, 5, 5, subsample=(2,2), border_mode="same"))
model.add(ELU())

model.add(Convolution2D(64, 5, 5, subsample=(2,2), border_mode="same"))
model.add(ELU()) 

model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode="same"))
model.add(ELU())

# flatten input to fully connected layer, 20% dropout
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())

# fully connected layer 1, 50% dropout
model.add(Dense(1024))
model.add(Dropout(.5))
model.add(ELU())

# fully connected layer 2
model.add(Dense(50))
model.add(ELU())

model.add(Dense(1))

# use adam optimizer as default per course advice, mean squared error loss function, learning rate 0.0001
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

# print model loaded to summarize structure and confirm layers are as expected 
print(model.summary())

# confirm information is loaded
print("input frames loaded: ", len(input_images))  

# wait to review
input("Hit enter to start training")

# training parameters
# batch size selected based on experience others had and memory constraints
# epochs increased manually varied until loss function stopped improving. 


batch_size = 32
nb_epoch = 30

# apply model to data using generator function
# store previous epochs (checkpoint function)
checkpointer = ModelCheckpoint(filepath="./weights/{epoch}.hdf5", verbose=1, save_best_only=False)

# run model training as defined
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=nb_epoch, callbacks=[checkpointer])

# save model to disk for playback
model.save("model.h5")

print("block completed")
