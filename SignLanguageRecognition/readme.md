# Sign Language Recognition
## _By Tuo Yang_

## Introductions
In this sign language recognition project, I created a sign detector, which detects numbers from 0 to 9 that can easily be extended to cover many other signs and hand gestures, including the alphabet.

## Dependicies
The prerequisite software and libraries for the sign project are:
-Python(3.7.4)
-IDE(Jupyter)
-Numpy (version 1.16.5)
-cv2 (openCV) (version 3.4.2) 
-Keras(version 2.3.1)
-Tensorflow(version 2.0.0)

## Implementations
Steps to Develop Sign Language Recognition Project
This project can be divided into three parts:

1. Creating the dataset
2. Training a CNN on the captured dataset
3. Predicting the data

## The File Architecutre of the Project
The files of this project in the current folder include:
Gesture - The folder of the datasets, which consists of two folders: train and test. Each folder contains the binary images collected from digits 0 to 9. Samples in the folder train are used for the training dataset (7020 images in total), while samples in the folder test are used for the validation dataset (410 images in total).
		  
- **Digits Recognition Results** - The trained CNN model's recognition results for digits hand gestures made in the living cam.

- **create_gesture.py** - The program file used for collecting samples of digits hand gestures in the living cam

- **Saved Model Files** - It is a folder containing following files:
    - **model_007_Adam.h5** - The best performance model trained with the Adam optimizer, with a validation accuracy of 94.63%
    - **model-020_SGD.h5** - The best performance model trained with the SGD optimizer, with a validation accuracy of 88.29%

- **readme.md** - Helping doc

- **Reference Gestures/reference_gesture.JPG** - Standard digit hand gesture used for reference

- **test.py** - The program file used for testing the model's performance

- **sign_language_recognition.ipynb** - The program file used for training the CNN model

- **gesture (stored on google drive: [link](https://drive.google.com/drive/folders/1RP5WrBVmJ5D3OFHSI_kDrCg1r4a1K0rW?usp=sharing))** - The folder contains the dataset used for training and testing.

- **recognition_results (stored on google drive: [link](https://drive.google.com/drive/folders/16K0ahu41RUzZb6iDaiDirbw9STzwnK2k?usp=sharing))** - The folder contains all digits recognition results in pictures,