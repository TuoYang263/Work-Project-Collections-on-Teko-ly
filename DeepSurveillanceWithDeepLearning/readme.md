# Deep Surveillance with Deep Learning - Intelligent Video Surveillance Project
## _By Tuo Yang_

## Objectives
This project aims to utilize deep learning algorithms to perform the video surveillance task using video sequence data collected by CCTV. The typical applications of deep surveillance are theft identification, violence detection,
detection of the chances of explosion.

## Network architecture
The deep neural network we use will be a 3-dimensional for learning spatio-temporal features of the video feed. 

For this video surveillance project, a spatio temporal autoencoder will be introduced, which is based on a 3D convolution network. The encoder part extracts the spatial and temporal information, and then the decoder reconstructs
the frames. The abnormal events are identified by computing the reconstruction loss using Euclidean distance between original and reconstructed batch. (More details about the network architecture can be checked in the image located
 at the same folder)
 
## Project Architecuture
- **train (store on google drive) [link](https://drive.google.com/drive/folders/1kNOnMSNPNlutQKdZGGpoLuM8jKC9UT9u?usp=sharing)**: The folder which contains all 50 video sequences used for training. 
- **train.py** - The program file used for training the model.
- **test.py** - The program file used for testing the trained model.
- **Testing Results** - It is a folder which includes:
    - **testing_demo_video.mp4**: Testing video, which is used for evaluating the model. 
- **Network Architecture** - It is a folder which includes:
    - **spatial-temporal-encoders.jpg**: The architecture of the network used in this project.
- **Saved Model Files** - It is a folder which includes:
    - **saved_model.h5** - The model file acquired from the training process
