# Color Detection
## _By Tuo Yang_

## Objectives
This project is for detecting colors through you click on a random pixel,
so there will be a data file containing the color name and its values.
Then we calculate the distance between each color and the pixel you click on
to identify which color this pixel is 

## Steps
This project is done by the following steps:

**Step 1. Taking an image from the user**
Use argparse library to create an argument parser, from which we can directly
give an image path from the command prompt.

**Step 2. Read the csv file with pandas**

**Step 3. Create the draw_function**
This fucntion will calculate the rgb value of when one pixel is clicked.
Parameters of this function include event name, (x, y) coordinates of 
the mouse-clicking position.

**Step 4. Calculate distance to get color name**

This function will return the color name of clicked pixel via r,g,and b values
acquired in the function draw_function. To get the color name, a distance will
be calculated to measure which color's rgb values have the minimum distance 
with ones taken from the pixel. The distance will be calculated by using the
Manhattan Distance, the formula of which is shown below:
    
d = abs(Red-ithRedColor) + abs(Green - ithGreenColor) + abs(Blue - ithBlueColor)

**Step 5.Display image on the window**
Inside a window, wherever a double click occurs, the color name and RGB values
will be updated correspondingly. 

Explainations for some opencv functions to be used in this part:
- cv2.imshow() - used for draw the image on the window
- cv2.rectangle(), cv2.putText() - used to draw a rectangle and get the color 
name to draw text on the window

## Project File Architecture

- **color_detection.py** - main program file 

- **colorpic.jpg** - the image used for color detection

- **colors.csv** - the data file storing various color info

- **program_running_ways.jpg** - display how to run the program in the command prompt

- **program_running_results.mp4** - videos recording program running results
