# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:15:20 2020

@author: david
"""

import cv2
import numpy as np

#Edge Detection Function
def canny(image):
    # Transform the image to gray scale for processing Hough Transform
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # apply smoothing by reducing noise, helps with edge detection
    blur = cv2.GaussianBlur(gray, (5,5),0)
    # Apply Canny method for edge detection
    canny = cv2.Canny(blur,50,150)
    return canny

# Plotting Region of Interest function
def region_of_interest(image):
    height = image.shape[0] # create an array for the image's pixesl
   # Create an array of polygons which we wish to map
    polygons = np.array([
            [(200, height),(1100, height),(550,250)]
            ])
    mask = np.zeros_like(image)         # mask the image 
    cv2.fillPoly(mask, polygons, 255)   # fill the mask with the polygon
    masked_image = cv2.bitwise_and(image, mask) # Detect edges for detecting region of interest
    return masked_image

# Plots the detected lines of the image
def display_lines(image,lines):
    line_image = np.zeros_like(image)   #create new black image array
    # We check if there are lines detected in the image
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2), (255,0,255), 10) # draw a line corresponding to line points
    return line_image

def make_coordinate(image, line_parameters):
    slope, intercept = line_parameters # store parameters into variables
    y1 = image.shape[0]                # grabs only the y values of image 
    y2 = int(y1*(3/5))                  
    x1 = int((y1 - intercept)/slope)   # grab the x values of image 
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])     
           
# Function for average slope of detected loines
def average_slope_intercept(image, lines):
    left_fit = []   # empty array for the left side of the lane
    right_fit = [] # empty array for the right side of the lane
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)    # put lane values into variables
        parameters = np.polyfit((x1,x2), (y1,y2), 1) # creating parameter for a linear function based on lines
        slope = parameters[0]       
        intercept = parameters[1]
        # store line values into the respective empty array
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # take the average of the line points
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinate(image, left_fit_average)
    right_line = make_coordinate(image, right_fit_average)
    return np.array([left_line, right_line])    
    

    
# Saving the image for processing
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)             # creating a copy of the image
canny_image = canny(lane_image)               # obtains filtered edge detection
cropped_image = region_of_interest(canny_image) # highlights region of interest
# We use Hough Transform to actively plot the lanes
lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# Average out the slops of lines for clear lane detection
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)   # Lane drawing
#Combined image of botht the image and the detected lanes
c_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)



cv2.imshow('result',c_image)              # Display the window
cv2.waitKey(0)                          # Press a key to exit window

