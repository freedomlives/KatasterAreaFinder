# MIT License

# Copyright (c) 2023 Andrew Alexander Ray

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import numpy as np
import random

# moments["m00"] is the area of the contour, make sure it isn't 0 before calling this function
def find_contour_center(contour):
    moments = cv2.moments(contour)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y


# Load the images
# map_image is the katasteral map
# I've only used this for E parcels so far, with green lines
# The number removal that is done to get clean map lines will be a problem
# when those lines are the same color as the text of the numbers.
map_image = cv2.imread('map_85011.png')
# red_image = the hranica uzivanie filled in red
# Obviously, these two images have to be exported from QGIS covering the same
# area and at the same resolution
red_image = cv2.imread("area_85011.png")


cleaned_image = map_image.copy() # The map without parcel numbers, eventually
original_image = map_image.copy() # Original map, to show parcel numbers in final output

##########################################################################################
# First part is removing the numbers from the katasteral map.
# There is a WMS source that doesn't put numbers on the map, but the lines it outputs
# are blurry, and I had trouble getting OpenCV to detect polygons on them.
##########################################################################################

# Convert the image to HSV
hsv_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for green color in HSV
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

# Create a mask for the green regions
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

# Change the color of green regions to white (255, 255, 255)
map_image[np.where(mask_green > 0)] = [255, 255, 255]

# Now convert it to gray
gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)

# Threshold the image to obtain binary (black and white) image
_, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)


##### These are here for debugging
# cv2.imshow('Threshold', thresholded)
# cv2.waitKey(0)


# Find contours in the binary image using RETR_TREE mode
contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

#####
#print(len(contours))

# Define the maximum area threshold for polygons (you can adjust this value)
# Needed to exclude the polygon that is the shape of the entire map
# Could use possible the length of the contour for this purpose as well- it is always 4
# Theoretically this could exclude a parcel, if the parcel would be perfectly square
max_area_threshold = 100000


### Information used when debugging
areas_list = []
for _, contour in enumerate(contours):
    areas_list.append(cv2.contourArea(contour))
mean = np.mean(areas_list)
median = np.median(areas_list)
print(f"Mean of areas: {mean}")
print(f"Median of areas: {median}")
height, width, channels = cleaned_image.shape # needed to check for out of bounds
print(f"Height: {height}")
print(f"Width: {width}")

# Will hold list of points inside the text to facillate erasing it
erase_cord_array = []

# Draw and fill each contour with a random color if its area is below the threshold
for idx, contour in enumerate(contours):
    # Calculate the area of the current contour
    area = cv2.contourArea(contour)
    ### print(area)

    # Approximate the contour with a simpler polygon (reduces number of vertices)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)


    # Draw and fill the contour (text) with black if its area is below the threshold
    if area < max_area_threshold:

        if area > 0:
            # Erasing by paint flooding the text to black, then paintflooding to white
            
            center_coords = find_contour_center(contour)

            cv2.drawContours(cleaned_image, [approx_polygon], -1, (0,0,0), thickness=cv2.FILLED)
            
            ### Used during debugging, may need again for C parcel debugging
            # if (cleaned_image[center_coords[1], center_coords[0]]==[0,0,0]).all():
            #     color=(255,0,0) # If center pixel is black, circle gets colored blue
            # else:
            #     color=(0,0,255) # Otherwise red
            # cv2.floodFill(cleaned_image, None, center_coords, (0,0,0))

            erase_cord_array.append(center_coords) 

            # cv2.circle(cleaned_image, center_coords, 2, color, 2)
    

### to check that the above worked
# Display the output image with the filled contours
# cv2.imshow('Text filled with black', cleaned_image)


# Now flood the fully black areas with white, checking first that the coordinate is actually black
#   (Since the coordinates represent digits of numbers, the first fill will turn the rest of area with number white)
for coords in erase_cord_array:
    if((cleaned_image[coords[1], coords[0]]==[0,0,0]).all()):
        cv2.floodFill(cleaned_image, None, coords, (255,255,255))


###############################################################################################
### Now with a clean map, we get down to the hardwork
###
###############################################################################################

# Define the font properties
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 0)  # Black text
thickness = 1
line_type = cv2.LINE_AA

colors_list = [] # To make visibly contrasting colors

# Read the property lines image (assuming you have already loaded it)
# cleaned_image = cv2.imread("map_small.png")

# Convert the image to grayscale
gray = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)


# cv2.imshow('gray', gray)
# cv2.waitKey(0)

# Threshold the image to obtain binary (black and white) image
_, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)


height, width, channels = cleaned_image.shape
# Define the size of the border (5 pixels)
border_size = 5

# Create a new image with increased dimensions for the border
# B/W image has no channels dimension!
thresholded_border = (
    np.ones((height + 2 * border_size, width + 2 * border_size), dtype=np.uint8) * 255
)
orig_border = (
    np.ones(
        (height + 2 * border_size, width + 2 * border_size, channels), dtype=np.uint8
    )
    * 255
)
red_border = (
    np.ones(
        (height + 2 * border_size, width + 2 * border_size, channels), dtype=np.uint8
    )
    * 255
)
nummap_border = (
    np.ones(
        (height + 2 * border_size, width + 2 * border_size, channels), dtype=np.uint8
    )
    * 255
)


# Copy the existing image onto the bordered image, leaving a border of white pixels
# The border is needed in the threshholded image so that none of the polygons are 
# at the edge of the image, as this seems to screw up OpenCV contour detection
# All other images that will be used then need the same border, easier than later
# translating coordinates :-)
thresholded_border[
    border_size : border_size + height, border_size : border_size + width
] = thresholded
orig_border[
    border_size : border_size + height, border_size : border_size + width
] = cleaned_image
red_border[
    border_size : border_size + height, border_size : border_size + width
] = red_image
## Nummap border - this is used to make the output images, it is the map with parcel numbers
nummap_border[
    border_size : border_size + height, border_size : border_size + width
] = original_image

# cv2.imshow("Threshold", thresholded_border)
# cv2.waitKey(0)

# Find contours in the binary image using RETR_TREE mode
contours, _ = cv2.findContours(
    thresholded_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1
)
print(len(contours))

# Create a copy of the original image to draw the filled contours on
output = cleaned_image.copy()

# Define the maximum area threshold for polygons (you can adjust this value)
max_area_threshold = 200000

# Draw and fill each contour with a random color if its area is below the threshold
for idx, contour in enumerate(contours):
    # Calculate the area of the current contour
    area = cv2.contourArea(contour)
    print(area)

    # Approximate the contour with a simpler polygon (reduces number of vertices)
    epsilon = 0.001 * cv2.arcLength(contour, True)  # The smaller, the more accurate
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Generate a random color (BGR format)
    # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if idx == 0:
        color = (0, 255, 0)
    else:
        color = (random.randint(0,255), random.randint(0,255), 0)
    colors_list.append(color)
    # Draw and fill the contour with the random color if its area is below the threshold
    if area < max_area_threshold:
        cv2.drawContours(orig_border, [approx_polygon], -1, color, thickness=cv2.FILLED)

# # Display the output image with the filled contours
# cv2.imshow("Filled Approximated Polygons (Up to Max Area)", orig_border)
# cv2.waitKey(0)

# Create image of map with field area overlaid
overlay = cv2.addWeighted(red_border,0.5,nummap_border, .5, 0)
overlay_source = overlay.copy()

percent_text = []
percent_coords = []

for idx, contour in enumerate(contours):
    mask = np.zeros_like(thresholded_border)
    # Draw the selected contour on the mask (in white)
    epsilon = 0.001 * cv2.arcLength(contour, True)  # The smaller, the more accurate
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(mask, [approx_polygon], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    # masked_image = cv2.bitwise_and(red_border, red_border, mask=mask)
    masked_image = np.zeros_like(orig_border)
    # red_border.copyTo(masked_image, mask)
    cv2.copyTo(src=red_border, mask=mask, dst=masked_image)

    # Convert the masked image to the HSV color space for easy color extraction
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a binary mask for the red pixels within the ROI
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Count the number of red pixels within the ROI
    num_red_pixels = np.count_nonzero(red_mask)

    # Count the total number of pixels within the ROI
    total_pixels_roi = np.count_nonzero(mask)

    # Calculate percentage
    percent_red = round(100 * (num_red_pixels / total_pixels_roi), 1)

    ### All of this print statements are just to help identify problems,
    ### They can be left out if everything is working
    print(f'Index {idx}')
    print(f"Red pixels: {num_red_pixels}")
    print(f"Total pixels: {total_pixels_roi}")
    print(f"Parcel area used: {percent_red}%")
    print(f"Contour size: {len(contour)}")
    print(f"Contour area: {cv2.contourArea(contour)}")
    print('------------')

    ### OpenCV finds random, tiny polygons. We want to keep them out of the final output text
    ### Also don't print percentages if it is 0%
    ### Note that two styles of output are being generated here
    if(cv2.contourArea(contour) > 0.0 and len(contour) > 4 and percent_red > 0.0 and total_pixels_roi > 25):
        text = f"{percent_red}"
        x, y = find_contour_center(contour)

        ####### These 2 arrays store the text to be also output in a different style later
        percent_text.append(text)
        percent_coords.append((x, y))
        ####### The rest of the code in this if statement could be removed, if only the final style is desired
        font_color = (255-colors_list[idx][0],0,0)
        font_color2 = (255-colors_list[idx][0],255,255)

        cv2.putText(
            orig_border,
            text,
            (x, y),
            font_face,
            font_scale,
            font_color,
            thickness,
            line_type,
        )

        cv2.rectangle(overlay, (x-5,y+5), (x+35, y-15), font_color, thickness=cv2.FILLED)

        cv2.putText(
            overlay,
            text,
            (x, y),
            font_face,
            font_scale,
            font_color2,
            thickness,
            line_type,
        )

    # if(total_pixels_roi > 200):
    #     cv2.imshow("Mask", mask)
    #     cv2.waitKey(0)

cv2.imshow("Percentages", orig_border)


cv2.imshow("Overlay", overlay)

###################################################################################
### And since neither of the above 2 outputs were totally satisfactory
### this prevents text from being written above other text and draws lines
### to point clearly to which parcel the percentage belongs
###################################################################################
font_thickness = 2
text_image = overlay_source.copy()
font_color = (255,255,255)

for text, (x, y) in zip(percent_text, percent_coords):
    # Find the position to place the text without overlapping with other texts
    text_width, text_height = cv2.getTextSize(text, font_face, font_scale, font_thickness)[0]
    text_x = max(0, min(x - text_width // 2 +20, text_image.shape[1] - text_width))
    text_y = max(text_height, min(y-20, text_image.shape[0] - text_height // 2))

    # Draw the text on the blank image
    cv2.putText(text_image, text, (text_x, text_y), font_face, font_scale, font_color, font_thickness)

    # Draw a line connecting the text to the coordinate
    cv2.line(text_image, (text_x + text_width // 2, text_y), (x, y), font_color, font_thickness)

# Apply alpha blending to overlay the text image on the original image
alpha = 0.6  # Adjust the transparency of the text image

###################################################################################
### TODO-  save dialog for output
###
###################################################################################
cv2.imshow('Percentages of each parcel used', text_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

