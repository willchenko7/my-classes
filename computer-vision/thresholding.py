import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('DATA/rainbow.jpg',0)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

#save threshold to tmp file
cv2.imwrite('tmp.jpg', thresh1)

#read in crossword image
img = cv2.imread('DATA/crossword.jpg',0)

cv2.imwrite('tmp.jpg', img)

#binary threshold
ret, thresh1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite('tmp.jpg', thresh1)

#adaptive threshold
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('tmp.jpg', thresh2)

#blend threshold images
blended = cv2.addWeighted(src1=thresh1, alpha=0.6, src2=thresh2, beta=0.4, gamma=0)
cv2.imwrite('tmp.jpg', blended)