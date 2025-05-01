# -*- coding: utf-8 -*-
"""
@author: Tuo Yang
In this program file, a bounding box will be created to detect ROI and calculate
the accumulated_avg as what was done in creating the dataset. This is for identifying
any foreground object.

Right now the max contour needs to be found, which is area of the detected hand.
Once the max contour is detected, the threshold of the ROI is taken as the test image.
The previous model trained by sign_language_recognition.ipynb will be used for prediction.
Its input is the ROI consisting of the hand
"""
# import required libraries
import numpy as np
import cv2
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import load_model

background = None
accumulated_weight = 0.5 # the accumulator forget 50% earliar images

# Load the model 
# Error record: ValueError: ('Unrecognized keyword arguments:', dict_keys(['ragged'])) while loading keras model
# Solution: https://stackoverflow.com/questions/60791067/valueerror-unrecognized-keyword-arguments-dict-keysragged-while-loa
model = load_model(r"E:/Python_Projects/MachineLearningProjects/SignLanguageRecognition/model_007_Adam.h5")

# Creating the regions for the ROI
ROI_left = 350     # the x coordinate of left-top corner in ROI 
ROI_top = 100      # the y coordinate of left-top corner in ROI
ROI_right = 150    # the x coordinate of bottom-right corner in ROI
ROI_bottom = 300   # the y coordinate of bottom-right corner in ROI

# word dictionary, it is used for querying to display text contents on the screen
word_dict = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five",
             6:"Six", 7:"Seven", 8:"Eight", 9:"Nine",}

"""
Calculate the accumulated weights of backgrounds
"""
def cal_accum_avg(frame, accumulated_weight):
    global background
    
    # if the variable background has no values, directly return the current frame 
    if background is None:
        background = frame.copy().astype("float")
        return None
    
    # This function calculates the weighted sum of the input image src and
    # the accumulator dst so that dst becomes a running average of a frame sequence
    # dst(x, y) <-- (1 - alpha) * dst(x, y) + alpha * src(x, y) if
    # mask(x, y) != 0
    # That is, alpha regulates the update speed (how fast the accumulator 
    # "forgets" about earlier images). The function supports multi-channel images.
    # Each channel is processed independently. 
    cv2.accumulateWeighted(frame, background, accumulated_weight)

"""
Calculate the threshold value for every frame and determine the contours 
using cv2.findContours and return the max contours (the most outermost contours
for the object) using the function segment.Using the contours we are able to
determine if there is any foreground object being detected in the ROI.
"""    
def segment_hand(frame, threshold=25):
    global background
    
    # Acquire abs values of pixels from background frame and current frame
    diff = cv2.absdiff(background.astype("uint8"), frame)
    
    # Get the binary image from the substracted image 
    _, thresholded =  cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    """
    Grab the external contours for the image
    This function retrieves contours from the binary image using the algorithm:
    Satoshi Suzuki and others. Topological structural analysis of digitized 
    binary images by border following. Computer Vision, Graphics, 
    and Image Processing, 30(1):32–46, 1985.
    
    The 1st parameter is the image to be processed
    The 2nd parameter is contour retrieval mode, cv2.RETR_EXTERNAL means
    retrieving only the extreme outer contours
    
    The 3rd parameter is contour approximation method, cv2.CHAIN_APPROX_SIMPLE
    means compressing horizontal, vertical, and diagonal segments and leaves
    only their end points. For example, an up-right rectangular contour is encoded
    with 4 points.
    """
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), 
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # if the number of contours is 0, return None
    if len(contours) == 0:
        return None
    else:
        # return the maximum contours with the largest area, which is the ROI
        # area, in other words, find connected components in the binary image
        hand_segment_max_cont = max(contours, key=cv2.contourArea)   
        return (thresholded, hand_segment_max_cont)
    
"""
Detect the hand now on the live cam feed
"""
# Open the laptop camera
cam = cv2.VideoCapture(0)

num_frames = 0            # the number of hand-gesture images for each digit

while True:
    # Read the current frame in the cam
    ret, frame = cam.read()
    
    # Flipping the frame to prevent inverted image of captured frame
    # mode: horizontal flip
    frame = cv2.flip(frame, 1)
    
    frame_copy = frame.copy() 
    
    # Extract ROI from the current frame
    roi = frame[ROI_top: ROI_bottom, ROI_right: ROI_left]
    
    # Convert the ROI to the gray-scale image
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Use GaussiasnBlur to make the ROI image blur
    # kernel size 9 by 9
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    
    # If the number of played frames is less than 60
    if num_frames < 60:
        # Calculate a running average of a frame sequence before the current frame
        cal_accum_avg(gray_frame, accumulated_weight)
        # fontScale: 1 thickness: 2
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", 
                    (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # if the number of played frame is greater than 60
    else:
        # Segment the hand region
        segmented_hand = segment_hand(gray_frame)
        
        # Check if the hand is detected
        if segmented_hand is not None:
            
            # Unpack the thresholded image and the max_contour
            thresholded, hand_segment = segmented_hand
            
            # Draw contours around hand segmentation results
            cv2.drawContours(frame_copy, 
                             [hand_segment + (ROI_right, ROI_top)],
                             -1, (255, 0, 0), 1)
            
            # Display the binary image after the segmentation
            cv2.imshow("Binary Hand Image", thresholded)
            # Convert the input image's size to fit for the input of
            # the CNN network. Resolution: 64 x 64 Number of channels: 3
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            # Add one more extra dimension to the image as the input of 
            # CNN network
            thresholded = np.reshape(thresholded, (1, thresholded.shape[0],
                                    thresholded.shape[0], 3))
            
            # Let the model make predictions for the thresholded image
            predictions = model.predict(thresholded)
            
            # Display the prediction results on the visual window
            cv2.putText(frame_copy, word_dict[np.argmax(predictions)],
                        (170, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            
        else:
            cv2.putText(frame_copy, "No hand detected...", (200, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Drawing ROI on frame copy
    # Taking advantage of two vertexes of backward slash to draw the rectangle
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), 
                  (ROI_right, ROI_bottom), (255, 128, 0), 3)
    
    cv2.putText(frame_copy, "Hand gesture recognition_ _ _", (10, 20),
                cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    
    # Increment the number of frames for tracking
    num_frames += 1
    
    # Display the frame with segmented hand 
    cv2.imshow("Sign Detection", frame_copy)  
    
    # Closing the window with the Esc key...(any other key with ord can be used too)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break
    
# Release the camera & destorying all the windows...
cv2.destroyAllWindows()
cam.release()



                          


