# Hand-written Digits Recognition
## _By Tuo Yang_

## Brief Introduction to This Project
A handwritten digit recognition app using the MNIST dataset will be implemented here. A simple CNN will be built to realize the recogition task. In the end, a GUI will be built, from which we can draw the digit and recognize it straight away.
 
## Project File Architecuture
- **model saved** : It is a folder which contains:
    - **mnist.h5**: The model file which stores self-customized CNN defined in model_building_and_training.py
- **evaluation results of model perforamance** - It is a folder which contains:
    - **digit 0,1,2...9.jpg**: Evaluation results of hand-written digits from GUI via running the script digits_recognizer_GUI.py
    - **model_building_and_training.py**: The script used for training self-customized models and save the trained model in the folder "model saved"
    - **digits_recognizer_GUI.py**: The script used for building digits recognition GUI to perform hand-written digits recognition tasks