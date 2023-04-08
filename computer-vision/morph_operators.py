import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, 'ABCDE', (50,300), font, 5, (255,255,255), 25)
    return blank_img

img  = load_img()

cv2.imwrite('tmp12.jpg', img)
print(img)

kernel = np.ones((5,5), np.uint8)
result = cv2.erode(img, kernel, iterations=4)
cv2.imwrite('tmp13.jpg', result)

white_noise = np.random.randint(low=0, high=2, size=(600,600))
#print(white_noise)
#save white noise image on grayscale
cv2.imwrite('tmp14.jpg', white_noise*255)
white_noise = white_noise * 255
noise_img = white_noise + img
cv2.imwrite('tmp15.jpg', noise_img)

opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
cv2.imwrite('tmp16.jpg', opening)

black_noise = np.random.randint(low=0, high=2, size=(600,600))
black_noise = black_noise * -255
black_noise_img = img + black_noise
black_noise_img[black_noise_img == -255] = 0
cv2.imwrite('tmp17.jpg', black_noise_img)

closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('tmp18.jpg', closing)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite('tmp19.jpg', gradient)