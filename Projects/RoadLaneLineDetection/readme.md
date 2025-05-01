# Road Lane Line Detection
## _By Tuo Yang_

## Road Lane Lines Detection Project

This project aims to identify road lines in which autonomous cars must run, which is a critical part of autonomous cars, as self-driving vehicles should not cross their lane and should not go in the opposite lane to avoid accidents.

## 1. Descriptions
The data for detecting road lane lines are video footage, including car-running scenes
with white and yellow solid road lane lines. For each frame of footage, road lane line detection
follows the steps below:

1) Convert the original image to gray-scale image gray1 and hsv image hsv1.
 
2) Use the color thresholding method to extract yellow and white lane lines respectively from hsv1 and gray1; after that, we can get binary mask images mask_yellow and mask_white of yellow and white lane lines.
   
3) Perform or operation to combine mask_yellow and mask_white to a new mask image mask_yw, then convert it
   back to gray image mask_yw_image.

4) Perform Gaussian blurring processing to mask_yw_image, then use the Canny operator to get edge detection image canny_edges.

5) Use Hough transform to detect multiple left and right lane lines from our interested ROI (Region of Interest). Now multiple lines are detected, we will filter out abnormal detected lines by setting up thresholds for slopes, then use the least square method to make fitness for multiple detected left and right lane lines to get the final results.
   
## 2. Getting started

1) Dependencies
   OS: Windows 10
   libaries: Python 3.9.4 + OpenCV 4.5.4
2) Program Running
   By running the script python line_detection.py, we can get the detection results of the road lane line in footage in the folder named road_lane_detection_output
   
   By running the script python GUI.py, we can get visualization results for road lane line detection in GUI.

```markdown ## 3. Project Organization
The project is organized as follows: 

├── README.md          				   <- The README for developers using this project.
├── image_processing_process           <- The folder includes images to display the whole image processing process for road lane line detection.
├── road_lane_detection_output         <- The detection results of road lane lines from all footages in the folder videos_input
├── videos_input                       <- The folder includes testing footage for detecting road lane lines.
├── detection_results_demo             <- A demo footage for showing running results of GUI.
├── line_detection.py                  <- Source code of building the detecting road lane lines method.
├── GUI.py             				   <- Source code used for building GUI to display results from running line_detection.py.
```