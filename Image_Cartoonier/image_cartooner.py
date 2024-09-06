# -*- coding: utf-8 -*-
"""
version: 1.0

@author: Tuo Yang
"""

"""
Importing the required modules
"""
import cv2         # imported to use OpenCV for image processing
import easygui     # imported to open a file box. It allow us to select any file from our system
import numpy as np # images are stored and processed as numbers. These are taken as arrays. We use numpy to deal with arrays
import imageio     # used to read the file which is chosen yb file box using a path
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

final_saving_path = 0
final_transformation_image = 0

"""
Using fileopenbox to open the box to choose file and help to store file as string
"""
def upload():
    global final_saving_path, final_transformation_image
    # return the path of the chosen file as a string
    image_path = easygui.fileopenbox()
    # the function used to read an image from a specified image path
    print(image_path)
    resized_image_v6 = cartoonify(image_path)
    final_saving_path = image_path
    final_transformation_image = resized_image_v6
 
"""
This function will perform multiple transformations. An image is converted into
a gray-scale image first. Then the gray-scale image is smoothened, edges of the
image will be extracted. Finally, a color image will be formed and masked with
eddges to get the cartoon image
"""
def cartoonify(image_path):
    # read the image
    original_image = cv2.imread(image_path)
    # reorder three color channels of an image to make it normally display
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # print(image)  # for display
    
    # confirm if an image is chosen
    if original_image is None:
        print("The image is not found, please choose it again")
        sys.exit()
    
    resized_image_v1 = cv2.resize(original_image, (960, 540))
    # plt.imshow(resized_image_v1, cmap='gray') # for display
    
    """
    Transforming an image to grayscale
    """
    # converting an image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    resized_image_v2 = cv2.resize(grayscale_image, (960, 540))
    # plt.imshow(resized_image_v2, cmap='gray') # for display
    
    """
    smoothening a gray-scale image
    """
    # applying median blur to smoothen an image
    # Here, the center pixel is assigned a mean value of all the pixels which
    # fall under the kernel, aiming for creating a blur effect
    smoothened_gray_scale = cv2.medianBlur(grayscale_image, 5)
    resized_image_v3 = cv2.resize(smoothened_gray_scale, (960, 540))
    # plt.imshow(resized_image_v3, cmap='gray') # for display
    
    """
    Retrieving the edges of an image 
    Explanation:
    Cartoon effect has two specialties: 1. Highlighted Edges 2.Smooth colors
    This will be done by adaptive thresholding technique. The threshold value
    is the mean of the neighborhood pixel values minus the constant C. C is 
    a constant that is subtracted from the mean or weighted sum of the
    neighborhood pixels. For the function adaptiveThreshold, the param 
    is the type of threshold applied, and the remaining parameters determine
    the block size
    """
    # Retrieving the edges for cartoon effect
    # by using thresholding technique
    get_edge = cv2.adaptiveThreshold(smoothened_gray_scale, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 9)
    resized_image_v4 = cv2.resize(get_edge, (960, 540))
    # plt.imshow(resized_image_v4, cmap='gray') # for display
    
    """
    Preparing a Mask Image
    The codes below will prepare the original lighted color image to mask with
    edges to produce a cartoon images. The bilateralFilter will be used to remove
    the noise (smoothening of an image to some degree)
    
    For the function bilateralFilter, the third parameter is the diameter of the
    pixel neighborhood, the fourth and fifth parameter defines sigmaColor and sigmaSpace.
    These parameters are used to give a sigma effect, making an image look vicious and like
    water paint and removing the roughness in colors
    """
    color_image = cv2.bilateralFilter(original_image, 9, 300, 300)
    resized_image_v5 = cv2.resize(color_image, (960, 540))
    # plt.imshow(resized_image_v5, cmap='gray') # for display
    
    """
    Giving a Cartoon Effect
    """
    # Masking the edged image with the processed color image
    cartoon_image = cv2.bitwise_and(color_image, color_image, mask=get_edge)
    resized_image_v6 = cv2.resize(cartoon_image, (960, 540))
    # plt.imshow(resized_image_v6, cmap='gray') # for display
    
    """
    Plotting all the transitions together
    """
    images = [resized_image_v1, resized_image_v2, resized_image_v3, resized_image_v4, 
             resized_image_v5, resized_image_v6]
    titles = ['original_image', 'gray_scale', 'smoothened_gray_scale', 'get_edge', 
             'mask_image', 'final_transformation_result']
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i], fontsize=17, color="black")
    plt.show()
    return resized_image_v6

"""
Define Functionally of save button
"""
def save(resized_image_v6, image_path):
    # saving an image using imwrite()
    new_name = "cartoonified_image"
    path_v1 = os.path.dirname(image_path)
    suffix = os.path.splitext(image_path)[1]
    final_saving_path = os.path.join(path_v1, new_name+suffix)
    cv2.imwrite(final_saving_path, cv2.cvtColor(resized_image_v6, cv2.COLOR_RGB2BGR))
    notice_info = "Image saved by name " + new_name + " at " + final_saving_path
    tk.messagebox.showinfo(title=None, message=notice_info)


if __name__=="__main__":
    """
    # Testify if cartoon transitions work
    image_path = 'testimonioLF.jpg'
    upload(image_path)
    """
    """
    Building the main window
    """
    top = tk.Tk()
    top.geometry('400x400')
    top.title('Cartoon Your Image')
    top.configure(background='white')
    lable = Label(top, background='#CDCDCD', font={'calibri', 20, 'bold'})
    
    """
    Making a Cartoonify button in the main window
    """
    upload = Button(top, text="Cartoonify an image", command=upload, padx=10, pady=5)
    upload.configure(background='#364156', foreground='white', font={'calibri', 10, 'bold'})
    upload.pack(side=TOP, pady=50)
    
    """
    Making a Save button in the main window
    """
    saving = Button(top, text="Save cartoon image",
                    command=lambda: save(final_transformation_image, final_saving_path), padx=30, pady=5)
    saving.configure(background='#364156', foreground='white', font={'calibri', 10, 'bold'})
    saving.pack(side=TOP, pady=50)
    
    """
    Main function to build the tkinter window
    """
    top.mainloop()

