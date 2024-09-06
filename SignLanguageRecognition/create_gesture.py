# -*- coding: utf-8 -*-
"""
Creating the dataset for sign language detection
There will be having a live feed from the video cam. every frame that detects
the hand in the ROI (Region of Interest) will be saved in the directory that
contains two folders train and test. Each containes 10 folders of captured images
using create_gesture_data.py

There will be two datasets created with this program file
For the train dataset, 701 images will be saved for each number to be detected
For the test dataset, 40 images will be saved for each number to be detected
@author: Tuo Yang
"""

# import required libraries


import warnings
import cv2
warnings.simplefilter(action="ignore", category=FutureWarning)

background = None
accumulated_weight = 0.5

# Creating the regions for the ROI
ROI_left = 350     # the x coordinate of left-top corner in ROI 
ROI_top = 100      # the y coordinate of left-top corner in ROI
ROI_right = 150    # the x coordinate of bottom-right corner in ROI
ROI_bottom = 300   # the y coordinate of bottom-right corner in ROI

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
When contours are detected, starting to save the image of the ROI in the train
and test set respectively for the number we are about to detect
"""
# Open the laptop camera
cam = cv2.VideoCapture(0)

num_frames = 0            # the number of hand-gesture images for each digit
element = 9               # the digit which the hand gesture will show, 
                          # Change the element 9 times to collect digital images 
                          # 9 times
num_imgs_taken = 0        # the number of images has been taken by the camera
target_imgs_num = 40      # the number of images about to be taken for one digit
saving_directory ="test"  # the directory used for saving images, two options available
                          # "train" or "test"

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
        if num_frames <= 59:
            # fontScale: 1 thickness: 2
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", 
                        (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # Configure the hand gesture in the ROI
    elif num_frames <= 300:
        # Segment the hand from the binary images
        segmented_hand = segment_hand(gray_frame)
        
        # The notice to remind the user of making hand gestures of digits
        cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), 
                    (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Check if the hand is actually detected by counting the number
        # of contours detected ...
        if segmented_hand is not None:
            thresholded, hand_segment = segmented_hand
        
            # Draw contours around the segmented image
            # Two parameters need attention:
            # 1.Thickness of lines the contours are drawn with, if it is negative,
            # the contours are drawn
            # 2.maxLevel: Maximal level for drawn contours, if it is 1, the function
            # draws the contours and all nested contours        
            cv2.drawContours(frame_copy, 
                             [hand_segment + (ROI_right, ROI_top)], # Do like this way??
                             -1, (255, 0, 0), 1)
            
            cv2.putText(frame_copy, str(num_frames) + " for " + str(element), (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Also display the binary image
            cv2.imshow("Thresholded Hand Image", thresholded)
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
            
            cv2.putText(frame_copy, str(num_frames), (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Text information showing how many images has been taken for the dataset
            cv2.putText(frame_copy, str(num_imgs_taken) + " images for " 
                        + str(element), (200, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the thresholded image
            cv2.imshow("Thresholded Hand Image", thresholded)
            # The upper limit for taking the images is 300
            if num_imgs_taken <= target_imgs_num:
                # Save the image to the specific directory
                cv2.imwrite(r"E:/Python_Projects/MachineLearningProjects/"+
                            "SignLanguageRecognition/gesture/"
                            + saving_directory + "/"
                            + str(element) + "/" +
                            str(num_imgs_taken+300) + '.jpg', thresholded)
            else:
                break
            num_imgs_taken += 1
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

                
            
            
            
        
            
            
        
    
        
    
    





