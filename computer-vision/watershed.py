import cv2
import numpy as np
import matplotlib.pyplot as plt

sep_coins = cv2.imread('DATA/pennies.jpg')

#save img to tmp file
cv2.imwrite('tmp/tmp51.jpg', sep_coins)

#convert to grayscale
gray_sep_coins = cv2.cvtColor(sep_coins, cv2.COLOR_BGR2GRAY)

#blur image
blur_sep_coins = cv2.medianBlur(gray_sep_coins, 25)

#binary threshold
ret, thresh = cv2.threshold(blur_sep_coins, 160, 255, cv2.THRESH_BINARY_INV)

#save img to tmp file
cv2.imwrite('tmp/tmp52.jpg', thresh)

#find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255,0,0), 10)

#save img to tmp file
cv2.imwrite('tmp/tmp53.jpg', sep_coins)


#use watershed

#load image
sep_coins = cv2.imread('DATA/pennies.jpg')

#convert to grayscale
gray_sep_coins = cv2.cvtColor(sep_coins, cv2.COLOR_BGR2GRAY)

#blur image
blur_sep_coins = cv2.medianBlur(gray_sep_coins, 35)

#binary threshold
ret, thresh = cv2.threshold(blur_sep_coins, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#save img to tmp file
cv2.imwrite('tmp/tmp54.jpg', thresh)

#noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#save img to tmp file
cv2.imwrite('tmp/tmp55.jpg', opening)

#apply distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

#save img to tmp file
cv2.imwrite('tmp/tmp56.jpg', dist_transform)

#apply threshold
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

#get sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

#save img to tmp file
cv2.imwrite('tmp/tmp57.jpg', sure_fg)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

#save img to tmp file
cv2.imwrite('tmp/tmp58.jpg', unknown)

#apply watershed
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0

markers = cv2.watershed(sep_coins, markers)

#find contours
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255,0,0), 10)

#save img to tmp file
cv2.imwrite('tmp/tmp59.jpg', sep_coins)