import cv2
import numpy as np
import matplotlib.pyplot as plt

dark_horse = cv2.imread('DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)
rainbow = cv2.imread('DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
blue_bricks = cv2.imread('DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

images = [dark_horse]
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([images[0]], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 50])
    plt.ylim([0, 100000])
#save histogram image
plt.savefig('tmp/tmp26.jpg')
channels = [0]
mask = None
histSize = [256]
ranges = [0, 256]
hist = cv2.calcHist(images, channels, mask, histSize, ranges)
print(hist.shape)
plt.plot(hist)
#save histogram image
plt.savefig('tmp/tmp27.jpg')