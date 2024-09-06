# -*- coding: utf-8 -*-
"""
Road Lane Line Detection Project
version: 1.0

This project aims to identify road lines in which autonomous cars must run, 
which is a critical part of autonomous cars, as self-driving cars should not
cross its lane and should not go in opposite lane to avoid accidents.

Steps to detect the road lane-line
1. For each frame in the video sequence, a mask should be created to mask
unnecessary pixels, which requires update those pixel values in the numpy
array of each frame
2. Use Hough Transform to detect mathematical shapes. Hough transformation
can detect shapes like rectangles, circles, triangles, and lines
"""

"""
Import required libraries
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
import matplotlib.image as mpimg
import math
from moviepy.editor import VideoFileClip

"""
Apply frame masking and find region of interest
"""
def interested_region(image, vertices):
    # Decide the mask color by judging if the image is the true color one
    if len(image.shape) > 2:
        # For the true color image, filling color (255, 255, 255) -> white
        mask_color_ignore = (255,) * image.shape[2]
    else:
        # Filling color of the gray-scale image
        mask_color_ignore = 255 # -> white
    
    mask = np.zeros_like(image)
    # cv2.fillpoly(filled_image, contours, filling_color)
    # The line below aims to create the mask of the current frame image
    cv2.fillPoly(mask, vertices, mask_color_ignore)
    # Use the created mask to perfrom the binary AND operation on each
    # pixel value of the image for getting the region of interest
    return cv2.bitwise_and(image, mask) # use and operation to keep roi

"""
Filter out outliers of all the detected lines 
"""
def filter_abnormal_lines(lines, slopes, threshold):
    while len(lines) > 0:
        mean = np.mean(slopes)                 # mean value of slopes
        diff = [abs(s - mean) for s in slopes] # calculate the slope difference of all detected road lane lines
        idx = np.argmax(diff)                  # find the index of element which has the biggest difference to others
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines

"""
Use the least square method to identify the unique road lane line
"""
def least_squares_fit(lines):
    # acquire x and y coordinates for making the 1D fitness
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # perfom the fitness of straight lines and get coefficients afterwards
    poly = np.polyfit(x_coords, y_coords, deg=1)
    # calculate two ends points of a line to identify this unique straight line
    # with polynomial coefficients
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int0)

"""
Create two lines in each frame after Hough transform
"""
def lines_drawn(image, lines, color=[255, 0, 0], thickness=6):
    global cache
    global first_frame
    slope_l, slope_r = [], []   # two sets used for collecting all lines' slopes
    lane_l, lane_r = [], []     # two sets used for collecting all lines' two ends'
                                # coordinates

    alpha = 0.2
    
    # if lines detection results are None, overlook the current frame
    if lines is None:
        return image
        
    # traverse all the detected lines from on the left and right sides of image 
    for line in lines:
        for x1, y1, x2, y2 in line:
            # use the detected line's two ends to get the slope
            slope = (y2 - y1)/(x2 - x1)
            # the line is on rightslide if its slope is greater than 0.4
            # otherwise it is on leftside
            if slope > 0.4:
                slope_r.append(slope)
                lane_r.append(line)  # line -> (x1, y1, x2, y2)
            elif slope < -0.4:
                slope_l.append(slope)
                lane_l.append(line)
        # image.shape[0] = min([y1, y2, image.shape[0]])
        
    # if no lines are detected, return
    if len(lane_l) == 0 or len(lane_r) == 0:
        print('no lane detected')
        return 1
    
    # take the mean values for all the collected slopes on the image both sides
    # , finally get slopes of two road lane lines on the road
    slope_mean_l = np.mean(slope_l, axis=0)
    slope_mean_r = np.mean(slope_r, axis=0)
    
    # get mean road lane lines on the left and right sides of the road
    mean_l = np.mean(np.array(lane_l), axis=0)
    mean_r = np.mean(np.array(lane_r), axis=0)
    
    # if two slopes do not exist or are infinity
    if np.isinf(slope_mean_r) or np.isinf(slope_mean_l):
        print('Dividing by zero')
        return 1
    
    # filter out abnormal lines
    l_thershold = 0.2
    r_threshold = 0.2
    
    left_lines = filter_abnormal_lines(lane_l, slope_l, l_thershold)
    right_lines = filter_abnormal_lines(lane_r, slope_r, r_threshold)
    
    left_lines = least_squares_fit(left_lines)
    right_lines = least_squares_fit(right_lines)
    
    x1_l, y1_l, x2_l, y2_l = left_lines[0][0], left_lines[0][1], left_lines[1][0], left_lines[1][1]
    
    x1_r, y1_r, x2_r, y2_r = right_lines[0][0], right_lines[0][1], right_lines[1][0], right_lines[1][1]
    
    # put all the detected points into an array
    present_frame = np.array([x1_l, y1_l, x2_l, y2_l, x1_r, y1_r, x2_r, y2_r], dtype="float32")
    
    # if it is the first frame, its road lane detection results are from current frame,
    # otherwise it will consider road lane detection results from previous frames
    if first_frame == 1:
        next_frame = present_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1 - alpha) * prev_frame + alpha * present_frame
    
    # draw these two detected road lanes on the current frame
    cv2.line(image, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
    cv2.line(image, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)
    
    # store current frame's road lane lines detection results
    cache = next_frame
    
    return image

"""
Conversion of pixels to a line in Hough Transform space
formula: x * cos(theta) + y * sin(theta) = rho
"""
def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    # Acquire lines in Hough Transform space through the function HoughLineP
    # return value lines would be multiple quadruples in format (x1, y1, x2, y2)
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    # Create a image which has the same size as the original image, and all pixels
    # are in black
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # Draw all acquired edge lines on the line_img. All line pixels are in white, 
    # while other pixels are in black 
    line_img = lines_drawn(line_img, lines)
    return line_img

"""
Function used for blending two images: img and initial_img (with the same size)
formula: dst = img * alpha + src * beta + gamma
"""
def weighted_img(img, initial_img, alpha = 0.8, beta = 1, lamb_da = 0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lamb_da)

"""
Process each frame of video to detect lane
"""
def process_image(image):
    global first_frame
    # plt.imsave('./raw_image.jpg', image)
    
    # the image is RGB
    # convert the image to grayscle and hsv ones
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # cv2.imwrite('./gray_image.jpg', gray_image)
    # cv2.imwrite('./hsv_image.jpg', hsv_image)
    
    # create a hsv color filtering range from orange yellow to light yellow,
    # sometimes the road lane line color in true color images is in yellow
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")
    
    # create a hsv image's mask, from which pixel value is 255 when its hsv pixel value
    # is between lower yellow and upper yellow, otherwise the pixel value would be 0
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    # acquire a gray-scale image mask of white road lane lines
    mask_white = cv2.inRange(gray_image, 200, 255)
    # combine road lane detection results in hsv and gray-scale images together
    # with or operation
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    # convert back the combined mask to gray-scale images 
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    
    """
    cv2.imwrite('./yellow_line_mask.jpg', mask_yellow)
    cv2.imwrite('./white_line_mask.jpg', mask_white)
    cv2.imwrite('./yellow_white_line_combined_mask.jpg', mask_yw)
    cv2.imwrite('./gray_image_mask.jpg', mask_yw_image)
    """
    
    # use guassian blur and canny edge detector to detect edges from images
    gaussian_sigmax = 1
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), gaussian_sigmax)
    # cv2.imwrite('./gauss_gray.jpg', gauss_gray)
    
    # threshold1: 50 threshold2: 150
    canny_edges = cv2.Canny(gauss_gray, 10, 150)
    
    # cv2.imwrite('./canny.jpg', canny_edges)
    
    # identify the region of interest image used for detecting road lane lines
    imshape = image.shape  # shape[0]: height shape[1]: width
    lower_left = [imshape[1]/9, imshape[0]]                                 # lower left corner coordiantes
    lower_right = [imshape[1] - imshape[1]/9, imshape[0]]                   # lower right corner coordiantes
    top_left = [imshape[1]/2 - imshape[1]/8, imshape[0]/2 + imshape[0]/10]  # top left corner coordinates
    top_right = [imshape[1]/2 + imshape[1]/8, imshape[0]/2 + imshape[0]/10] # top right corner coordinates
    
    # build vertices along rectangle edges
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    
    # extract roi from Canny edge images
    roi_image = interested_region(canny_edges, vertices)
    # cv2.imwrite('./roi.jpg', roi_image)
    
    theta = np.pi/180
    
    # parameters explanations: image, rho, theta, threshold, min_line_len, max_line_gap
    line_image = hough_lines(roi_image, 1, theta, 30, 150, 100)
    # cv2.imwrite('./line_image.jpg', line_image)
    
    # blend the original image and line image together with weights
    result = weighted_img(line_image, image, alpha=0.8, beta=1., lamb_da=0.)
    # plt.imsave('./final_result.jpg', result)
    return result

"""
There are three demo videos could be used for test:
1. challenge.mp4: with solid yellow and white lane lines on road both sides, and the car is driving on the bend lane 
2. solidYellowLeft.mp4: with soild yellow road lane lines on the left side, and the car is running on the straight lane
3. solidWhiteRight.mp4: with soild white road lane lines on the right side, and the car is running on the straight lane
"""
first_frame = 1
project_root_path = 'E:/Python_Projects/MachineLearningProjects/RoadLaneLineDetection/'
white_output = project_root_path + 'road_lane_detection_output/solidYellowLeft_results.mp4'  # path to output file 
clip1 = VideoFileClip(project_root_path + 'videos_input/solidYellowLeft.mp4')         # path to input file
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
        
    
