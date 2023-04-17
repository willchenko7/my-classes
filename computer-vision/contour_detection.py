import cv2
import numpy as np
import matplotlib.pyplot as plt

#read in internal_extrnal
img = cv2.imread('DATA/internal_external.png',0)

#save img to tmp file
cv2.imwrite('tmp/tmp44.jpg', img)

#find contours
contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(img.shape)
#draw contours
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)

#save external_contours to tmp file
cv2.imwrite('tmp/tmp45.jpg', external_contours)

internal_contours = np.zeros(img.shape)
#draw contours
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)

#save internal_contours to tmp file
cv2.imwrite('tmp/tmp46.jpg', internal_contours)