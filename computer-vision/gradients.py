import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('DATA/sudoku.jpg', 0)
cv2.imwrite('tmp/tmp20.jpg', img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
cv2.imwrite('tmp/tmp21.jpg', sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
cv2.imwrite('tmp/tmp22.jpg', sobely)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
cv2.imwrite('tmp/tmp23.jpg', laplacian)

canny = cv2.Canny(img, 20, 170)
cv2.imwrite('tmp/tmp24.jpg', canny)

blended = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0)
cv2.imwrite('tmp/tmp25.jpg', blended)

ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('tmp/tmp26.jpg', thresh)