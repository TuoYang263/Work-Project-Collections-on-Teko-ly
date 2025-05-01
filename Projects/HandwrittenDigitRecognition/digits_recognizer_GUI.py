# -*- coding: utf-8 -*-
"""
version: 1.0

@author: Tuo Yang
"""
"""
6. Create GUI to predict digits 
Building an interactive window to draw digits on canvas,

  (1).Among this part of codes, the function predict_digit() takes
  the image as input and then uses the trained model to predict
  the digit.

  (2).The App class is responsible for building the GUI, a canvas
  would be created inside for capturing the mouse event trigged by a
  button, once the event is trigged, the function predict_digit()
  will be executed and display prediction results
"""
from keras.models import load_model
from tkinter import * 
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps
import numpy as np
import matplotlib.pyplot as plt

model = load_model('C:/Projects to deal with/HandwrittenDigitRecognition/model saved/mnist.h5')

def predict_digit(img):
    print(img.size)
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img/255.0
    plt.imshow(img[0, :, :, 0], cmap='gray')
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.x = self.y = 0
        
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white",
                                cursor="cross")
        self.label = tk.Label(self, text="Thinking...", 
                              font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize",
                                      command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", 
                                      command=self.clear_all)
        
        # Grid structure: the location settings of all window components
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, padx=2, pady=2)
        self.classify_btn.grid(row=1, column=1, padx=2, pady=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        # Bind mouse event with the button component, the binding event
        # <B1-Motion> is triggered by the left mouse button
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    
    def clear_all(self):
        self.canvas.delete("all")  # clear out contents in the canvas
    
    def classify_handwriting(self):
        # get the handle of the canvas
        canvas_handle = self.canvas.winfo_id()
        # get the coordinate of the canvas
        rect = win32gui.GetWindowRect(canvas_handle)
        grabbed_image = ImageGrab.grab(rect)
        
        digit, acc = predict_digit(grabbed_image)
        print('digit:', str(digit), ' accuracy:', str(acc))
        self.label.configure(text=str(digit) + ', ' + str(int(acc*100))
                             + '%')
    
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8  # decide the width of lines
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, 
                                self.y+r, fill='black')

app = App()
mainloop() # display GUI, waiting for response of events
        
        
        
        
