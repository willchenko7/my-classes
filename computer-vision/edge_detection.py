import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DATA/sammy_face.jpg')

#save img to tmp file
cv2.imwrite('tmp/tmp37.jpg', img)

edges = cv2.Canny(image=img, threshold1=127, threshold2=127)

#save edges to tmp file
cv2.imwrite('tmp/tmp38.jpg', edges)


med_val = np.median(img)

# Lower threshold to either 0 or 70% of the median value, whichever is greater
lower = int(max(0, 0.7 * med_val))
# Upper threshold to either 130% of the median or the max 255, whichever is smaller
upper = int(min(255, 1.3 * med_val))

edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)

#save edges to tmp file
cv2.imwrite('tmp/tmp39.jpg', edges)

blurred_img = cv2.blur(img, ksize=(5,5))

#get edges from blurred image
edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)

#save edges to tmp file
cv2.imwrite('tmp/tmp40.jpg', edges)