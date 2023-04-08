import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DATA/bricks.jpg')

#cv2.imwrite('tmp8.jpg', dst)

#gamma = 1/4
#np.power(img, gamma)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'bricks', (10,600), font, 10, (255,0,0), 4, cv2.LINE_AA)

#blurred = cv2.blur(img, (10,10))
blurred = cv2.GaussianBlur(img, (5,5), 10)
cv2.imwrite('tmp8.jpg', blurred)

#create kernel for blurring
#kernel = np.ones((5,5), np.float32)/25
#dst = cv2.filter2D(img, -1, kernel)
#cv2.imwrite('tmp8.jpg', dst)

img = cv2.imread('DATA/sammy.jpg')
cv2.imwrite('tmp9.jpg', img)
noise_img = cv2.imread('DATA/sammy_noise.jpg')
#convert noise image to RGB
noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('tmp10.jpg', noise_img)
median = cv2.medianBlur(noise_img, 5)
cv2.imwrite('tmp11.jpg', median)