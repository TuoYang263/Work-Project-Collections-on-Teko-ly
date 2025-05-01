# Real-Time Face Mask Detector
## _By Tuo Yang_

## Introduction
This project will implement a real-time system to detect if the person showing up on the webcam is wearing a mask, and the face mask detector model will be trained with Keras and OpenCV. The whole project's implementation has two parts:
1. Using Keras in a Python script to train the face detector model and save the model that performs the best.
2. Using OpenCV and haar cascade face detection to visualize mask detection results in a real-time webcam.
 
## Project File Architecture
- **evaluation_results**: It is a folder which contains:
    - **demo_video.mp4**: Video footage that displays how the mask detector model performs in front of the webcam.
- **haarcascade_frontalface_default.xml**: Model files used for constructing haar cascade features for face detection.
- **mask_detector.ipynb**: The program file is used to train the dataset and save the best model in the file.
- **test.py**: The program file used for evaluating the model, reading frames from the internal laptop cam to detect if the detected object in the shot wears the mask.
- **datasets (stored google drive: [link](https://drive.google.com/drive/folders/1bh2YTyovbSEwp4TiQn6KKdcoa4RPxr1S?usp=sharing))**: It is a folder which contains:
    - **train**: The training dataset contains 50 photos with masks and 50 without masks.
    - **test**: The testing dataset contains 50 photos with masks and 50 without masks.
- **model_files (stored google drive: [link](https://drive.google.com/drive/folders/13nN9hgeNShRVlQX8COZz74AvENSAYBWe?usp=sharing))**: It is a folder which contains:
    - **model2-019.h5**: The saved model file performs best during training.