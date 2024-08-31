"""
Author: Jackson Snyder
NetID: jsnyde10
This algorithm is designed to find waldo!
If it finds something that it thinks is waldo then it will 
return the original image with green boxes around any potential hits
"""

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')


def waldo_recognition_algorithm(wheres_waldo):
    # proscesses input image, returns number of hits and image with bounding boxes
    # create a red mask within desired range
    lower_red = (150, 0, 0)
    upper_red = (225, 80, 100)
    red_mask = cv2.inRange(wheres_waldo, lower_red, upper_red)

    # change color type, apply gausian blur, and detect edges
    gray_waldo = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    blurred_waldo = cv2.GaussianBlur(gray_waldo, (5, 5), 0)
    edges_waldo = cv2.Canny(blurred_waldo, 50, 150, apertureSize=3)

    # detect lines in mask
    lines_waldo = cv2.HoughLinesP(edges_waldo, 1, np.pi/180, 60, minLineLength=25, maxLineGap=10)
    waldo_img_lines = np.copy(wheres_waldo)
    hits = 0
    for line in lines_waldo:
        x1, y1, x2, y2 = line[0]

        # checks to see if each detected line in horizontal and of required length
        if y2 - y1 <= 2 and 5 <= x2 - x1 <= 30:

            # if line is good, create bounding box parameters based on line dimensions
            left = min(x1 - 25, x2 - 25)
            top = min(y1 + 60, y2 + 60)
            right = max(x1 + 25, x2 + 25)
            bottom = max(y1 - 50, y2 - 50)

            # create new array of the box defined above and inital image within box
            box_rgb = cv2.cvtColor(wheres_waldo, cv2.COLOR_BGR2RGB)
            box = box_rgb[bottom:top, left:right]

            # creat brown mask to detect hair
            # uses np.logical over cv2.inrange because the goal is to find the number of pixels, not create a new image
            brown_lower = np.array([20, 10, 10])
            brown_upper = np.array([35, 15, 15])
            hair_mask = np.logical_and.reduce((brown_lower <= box, box <= brown_upper))

            # if more than 750 pixels of hair detected on box, print box onto copy of initial image
            count = np.sum(hair_mask)
            if count > 750:
                cv2.rectangle(waldo_img_lines, (left, top), (right, bottom), (0, 255, 0), 3)
                print("Possible Waldo found!")
                hits += 1
    return hits, waldo_img_lines


def main():
    print("Welcome to the Waldo finder!\nThis algorithm will try to find waldo in your provided image\n")

    # request image name from user, imports the image into an array
    try:
        img_name = input(r"Enter name of image: ")
        wheres_waldo = np.asarray(Image.open(img_name))
    except IOError:
        print("Image file cannot be found, exiting now")
        quit()

    print("\nComputing......")
    hits, waldo_img_lines = waldo_recognition_algorithm(wheres_waldo)
    
    # show image plus bounding box in new window
    if hits > 0:
        plt.imshow(waldo_img_lines)
        plt.show()
    else:
        print("Sorry, I could not seem to find anything.")


if __name__ == '__main__':
    main()
