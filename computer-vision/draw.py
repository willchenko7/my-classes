import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
This script is used to show how to use cv2 to draw on images
'''

#create a black image
blank_img = np.zeros((512,512,3), np.int16)

#save to tmp file
cv2.imwrite('tmp.jpg', blank_img)

#draw a rectangle
cv2.rectangle(blank_img, pt1=(384,0), pt2=(510,150), color=(0,255,0), thickness=10)

#draw a rectangle in the middle
cv2.rectangle(blank_img, pt1=(200,200), pt2=(300,300), color=(0,0,255), thickness=10)

#draw a circle
cv2.circle(blank_img, center=(100,100), radius=50, color=(255,0,0), thickness=8)

#draw a circle filled in
cv2.circle(blank_img, center=(400,400), radius=50, color=(255,0,0), thickness=-1)

#draw purple line from top left to bottom right
cv2.line(blank_img, pt1=(0,0), pt2=(512,512), color=(255,0,255), thickness=5)

#draw a font
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img, text='Hello', org=(10,500), fontFace=font, fontScale=4, color=(255,255,255), thickness=3, lineType=cv2.LINE_AA)

#create custom polygon
vertices = np.array([[100,300], [200,200], [400,300], [200,400]], np.int32)
pts = vertices.reshape((-1,1,2))
cv2.polylines(blank_img, [pts], isClosed=True, color=(0,255,255), thickness=5)
