# -*- coding: utf-8 -*-
"""
The test program file of the deep surveillance system
version: 1.0

@author: Tuo Yang
"""

# Import required libraries
import cv2
import numpy as np
import imutils
from keras.models import load_model

# Build the function to compute mean_squared_loss
def mean_squared_loss(x1, x2):
    # Calculate the difference between every dimension of the data
    difference = x1 - x2
    # Acquire the shape of the difference
    dim1, dim2, dim3, dim4, dim5 = difference.shape
    samples_num = dim1 * dim2 * dim3 * dim4 * dim5
    sequence_difference = difference ** 2
    summation = sequence_difference.sum()
    final_distance = np.sqrt(summation)
    mean_distance = final_distance/samples_num
    return mean_distance

# Load the model
model = load_model("saved_model.h5")
# Load the video for testing
cap = cv2.VideoCapture("./fighting_scene1.mp4")
print(cap.isOpened())
# The number of frames to be played
frame_num = 10

while cap.isOpened():
    # The variable used for storing image frames
    image_dump = [] 
    # Read every frame into the variable and check if every frame is successfully read
    ret, frame = cap.read()
    
    # Read 10 frames for prediction first
    for i in range(frame_num):
        ref, frame = cap.read()
        image = imutils.resize(frame, width=700, height=600)
        
        # Resize each frame into a specific size (227, 227) by resampling using
        # pixel area relation
        frame = cv2.resize(frame, (227, 227), interpolation = cv2.INTER_AREA)
        # Convert the each true color frame to gray scale images
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] +\
                0.1140 * frame[:, :, 2]
        # Standardize each frame
        gray = (gray - gray.mean())/gray.std()
        # Restrict each pixel's values within [0, 1]
        gray = np.clip(gray, 0, 1)
        image_dump.append(gray)
        
    # Convert image_dump to the numpy array
    image_dump = np.array(image_dump)
    
    image_dump.resize(227, 227, frame_num)
    
    # Resize the numpy array, expanding the dimension at the 1st and 5th dimension
    image_dump = np.expand_dims(image_dump, axis=0)
    image_dump = np.expand_dims(image_dump, axis=4)
    
    output = model.predict(image_dump)
    
    loss = mean_squared_loss(image_dump, output)
    print(loss)
    
    if frame.any() == None:
        print("None")
    
    # If quitting the system, press ESC or q
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    if loss > 0.00074:
        #print('Abnormal Event Detected')
        cv2.putText(image, "Abnormal Event Detected", (150, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("video", image)

cap.release()
cv2.destroyAllWindows()
        
    
    
