# -*- coding: utf-8 -*-
"""
Version: 1.0

@author: Tuo Yang

This project is for detecting colors through you click on a random pixel,
so there will be a data file containing the color name and its values.
Then we calculate the distance between each color and the pixel you click on
to identify which color this pixel is 
"""

"""
1. Taking an image from the user
Use argparse library to create an argument parser, from which we can directly
give an image path from the command prompt.
"""
import argparse
import cv2
import pandas as pd

# Declare some global variables
r = None
g = None
b = None
xpos = None
ypos = None
clicked = None

# Creat an order with options in the command prompt
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
# Get the image path from the command
img_path = args['image']
# Reading the image with opencv
img = cv2.imread(img_path)

"""
2.Read the csv file with pandas
"""
# Read csv file with pandas and give names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

"""
3. Create the draw_function
This fucntion will calculate the rgb value of when one pixel is clicked.
Parameters of this function include event name, (x, y) coordinates of 
the mouse-clicking position.
"""
def draw_function(event, x, y, flags, param):
    # if the clicking event of the left button happens,
    # opertations below will be performed
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global r, g, b, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x] # this depends on the coordinates system of the image
        r = int(r)
        g = int(g)
        b = int(b)

"""
4. Calculate distance to get color name
This function will return the color name of clicked pixel via r,g,and b values
acquired in the function draw_function. To get the color name, a distance will
be calculated to measure which color's rgb values have the minimum distance 
with ones taken from the pixel. The distance will be calculated by using the
Manhattan Distance, the formula of which is shown below:
    
d = abs(Red-ithRedColor) + abs(Green - ithGreenColor) + abs(Blue - ithBlueColor)
"""
def getColorName(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        distance = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"]))\
                   + abs(B - int(csv.loc[i, "B"]))
        if distance <= minimum:
            minimum = distance
            color_name = csv.loc[i, "color_name"]
    return color_name

"""
5.Display image on the window
Inside a window, wherever a double click occurs, the color name and RGB values
will be updated correspondingly. 
Explainations for some opencv functions to be used in this part:
cv2.imshow() - used for draw the image on the window
cv2.rectangle(), cv2.putText() - used to draw a rectangle and get the color 
name to draw text on the window
"""
while True:
    cv2.imshow("image", img)
    # Bind the mouse click event with the window together
    cv2.setMouseCallback('image', draw_function) 
    print("Clicked:", clicked)
    if clicked:
        # Parameters settings for cv2.rectangle()
        # cv2.rectangle(image, startpoint, endpoint, color, thickness)
        # Here the thickness is set as -1 to fill the rectangle entirely
        # The color used to fill in is the color from the clicked pixel
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
        
        # Create textual strings for display (Color name and RGB values)
        text = getColorName(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B='\
            + str(b)
        
        # Parameters settings for cv2.putText()
        # cv2.putText(img, text, start, font(0-7), fontScale, color, thickness,
        # lineType, (optional bottomLeft bool)))
        # If the color of the clicked pixel is light, displaying font color
        # will be black, otherwise it will be white
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if r + g + b >= 600:  # if the color of clicked pixel is light
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        clicked = False
        
    # Break the loop when user hits 'esc' key 
    if cv2.waitKey(20) & 0XFF == 27:
        break

# Jump out of the loop, exit out all the windows
cv2.destroyAllWindows()
        








