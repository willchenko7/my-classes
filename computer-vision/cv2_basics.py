import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
This script is used to show how to use cv2 to read, write, resize, flip images
'''

# Read image
img = cv2.imread('DATA/00-puppy.jpg')

#convert from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#save image to a tmp file
cv2.imwrite('tmp.jpg', img_flipped)

#read image as grayscale
img_gray = cv2.imread('DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)

#resize image
img_resized = cv2.resize(img, (1000,400))

#resize image by ratio
w_ratio = 0.5
h_ratio = 0.5
img_resized = cv2.resize(img, (0,0), img, w_ratio, h_ratio)

#flip image
img_flipped = cv2.flip(img, 0)

