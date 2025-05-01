# -*- coding: utf-8 -*-
"""
Deep Surveillance with Deep Learning - Intelligent Video Surveillance Project
Version: 1.0

@author: Tuo Yang

This project aims to utilize deep learning algorithms to perform the video 
surveillance task using video sequence data collected by CCTV. The typical
applications of deep surveillance are theft identification, violence detection,
detection of the chances of explosion.

Network architecture
The deep neural network we use will be a 3-dimensional for learning 
spatio-temporal features of the video feed. 

For this video surveillance project, a spatio temporal autoencoder will be
introduced, which is based on a 3D convolution network. The encoder part
extracts the spatial and temporal information, and then the decoder reconstructs
the frames. The abnormal events are identified by computing the reconstruction
loss using Euclidean distance between original and reconstructed batch.
(More details about the network architecture can be checked in the image located
 at the same folder)
"""

"""
1. Import required libraries
"""
import numpy as np
import glob
import os
import cv2
import imutils
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

"""
2. Initialize directory path variable and describe a function to process
and store video frames
"""
# The variable used for storing image frames
store_image = []
# The directory where the training video sequence is located at 
train_path = './train'
# Video playing speed: 5 fps
fps = 5
# Create the path of all training frames
train_images_path = train_path + '/frames'
# Return names of all files and folders under the directory train_path
train_videos = os.listdir(train_images_path)


def store_frames_inarray(image_path):
    # Load the video sequence from the image_path
    image = load_img(image_path)
    # Convert all loaded frames to numpy arrays
    image = img_to_array(image)
    # Resize each frame into a specific size (227, 227) by resampling using
    # pixel area relation
    image = cv2.resize(image, (227, 227), interpolation = cv2.INTER_AREA)
    # Convert the each true color frame to gray scale images
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] +\
           0.1140 * image[:, :, 2]
    # Add all converted frames to the list store_image 
    store_image.append(gray)
    
"""
3. Extract frames from video and call store function
"""
for index, video in enumerate(train_videos):
    #os.system('ffmpeg -i {}/{} -r 1/{} {}/frames/Train00'.format(train_path, video, fps, train_path) 
              #+ str(index + 1) + '/%03d.tif')
    video_frames = os.listdir(train_images_path + '/'+ video + '/')
    for frame in video_frames:
        frame_path = train_images_path + '/' + video + '/' + frame
        store_frames_inarray(frame_path)
        
"""
4. Store the store_image list in a numpy file "training.npy"
"""
# Convert the store_image to the numpy array
store_image = np.array(store_image)
# The order of outputs for store_image.shape (frames_num, height, width)
frame_num, frame_height, frame_width = store_image.shape

# Resize the store_image
store_image.resize(frame_height, frame_width, frame_num)
# Perform the standardlization to the video frames
store_image = (store_image - store_image.mean()) / (store_image.std())
# Restrict all pixel values in store_image within [0, 1] (Restricted maximum 
# and minimum values)
store_image = np.clip(store_image, 0, 1)
np.save('training.npy', store_image)

"""
5. Create spatial autoencoder architecture
"""
spatial_model = Sequential()

spatial_model.add(Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1),
                         padding='valid', input_shape=(227, 227, 10, 1), 
                         activation='tanh'))
spatial_model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1),
                         padding='valid', activation='tanh'))
spatial_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1,
                             padding='same', dropout=0.4, recurrent_dropout=0.3,
                             return_sequences=True))
spatial_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1,
                             padding='same', dropout=0.3, return_sequences=True))
spatial_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1,
                             padding='same', dropout=0.5, return_sequences=True))
spatial_model.add(Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1),
                         padding='valid', activation='tanh'))
spatial_model.add(Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1),
                         padding='valid', activation='tanh'))

spatial_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

"""
6. Train the autoencoderl on the "training.npy" file and save the model
with name "saved_model.h5"
"""
# Load the training data stored in npy files
training_data = np.load('training.npy')
# Acquire the number of all frames 
frames = training_data.shape[2]
# Take 10 frames as a unit, filtering out the last few frames
frames = frames - frames % 10

# Store filtered frames to the variable
training_data = training_data[:, :, :frames]
training_data = training_data.reshape(-1, 227, 227, 10)
# Expand another dimension to the training data for inputs of network
training_data = np.expand_dims(training_data, axis=4)
target_data = training_data.copy()

epochs = 5
batch_size = 1

# Set up two callbacks during the training
callback_save = ModelCheckpoint("saved_model.h5", monitor="mean_squared_error",
                                save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

spatial_model.fit(training_data, target_data, batch_size=batch_size, epochs=epochs,
                  callbacks = [callback_save, callback_early_stopping])
spatial_model.save("saved_model.h5")




    


    
    



