# -*- coding: utf-8 -*-
"""
Version: 1.0 

@author: Tuo Yang
"""
import cv2
import numpy as np
from keras.models import load_model
# load trained model from model files
model = load_model("./model2-019.h5")

# the dictionary used for storing two types of results
# 0 - without wearing the mask, 1 - wearing the mask
results = {0:'without mask', 1:'mask'}

# the color dictionary used for indicating if the detected object
# wears the mask or not. (0, 0, 255) - blue, represents not wearing the mask
# (0, 255, 0) - light green, represents wearing the mask
GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Reduction factor of the size of a frame
rect_size = 4
# Open the internal cam of the laptop
cap = cv2.VideoCapture(0)

# Use xml file haarcascade_frontalface_default.xml from opencv libraries
# to detect humanity faces, programs inside this file build cascade classifiers
# based on Haar features
haarcascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


while True:
    # Capture the image of one frame
    # two return values: 
    # rval: indicates if the image is read or not 
    # frame: stores the image of one frame
    (rval, frame_image) = cap.read()
    # filp function, code 1 represents horizontal flip
    frame_image = cv2.flip(frame_image, 1, 1)
    
    # Resize the frame captured by t)he internal cam
    resized_frame = cv2.resize(frame_image, (frame_image.shape[1] // rect_size, 
                              frame_image.shape[0] // rect_size))
    # Multi-scale faces are detected with using the function below
    faces = haarcascade.detectMultiScale(resized_frame)
    # Traverse information of all detected faces
    for face in faces:
        # There will be a quaternion information returned back, 
        # which represents the coordinates info of detected faces.
        # However, since these coordinates acquired from resized frames,
        # they should multiple with rect_size to resume back their actual
        # coordinates in the original frame
        (x, y, width, height) = [v * rect_size for v in face]
        
        # Extract the face's image from the current originally-sized frame 
        face_img = frame_image[y:y+height, x:x+width]
        # Codes below will perform processing to extracted images for making
        # them can be inputs of the network (make predictions). 
        resized_face = cv2.resize(face_img, (300, 300))
        normalized_face = resized_face/255.0
        reshaped_face = np.reshape(normalized_face, (1, 300, 300, 3))
        # ???
        reshaped_face = np.vstack([reshaped_face])
        result = model.predict(reshaped_face)
        print(result)
        
        # Making comparison across rows, return the index of element which
        # is the maximum of per row
        label = np.argmax(result, axis=1)[0]
        
        # The visualization of mask wearing detection results,
        # which is combined by three parts: 
        # 1. The rectangle above the detection box
        # 2. Notice info: mask or without mask
        # 3. The rectangle which draw the face detection results
        cv2.rectangle(frame_image, (x, y), (x+width, y+height), 
                      GR_dict[label], 2)
        cv2.rectangle(frame_image, (x, y-40), (x+width, y), 
                      GR_dict[label], -1)
        cv2.putText(frame_image, results[label], (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    cv2.imshow('LIVE', frame_image)
    key = cv2.waitKey(10)

    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
    

    
    
    
    
    
    

