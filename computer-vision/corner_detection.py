import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('DATA/flat_chessboard.png')

#save flat_chess to tmp file
cv2.imwrite('tmp/tmp31.jpg', flat_chess)

real_chess = cv2.imread('DATA/real_chessboard.jpg')
#save real_chess to tmp file
cv2.imwrite('tmp/tmp32.jpg', real_chess)

#apply haris corner detection
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_RGB2GRAY)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)

dst = cv2.cornerHarris(src=gray_flat_chess, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

flat_chess[dst>0.01*dst.max()] = [255,0,0]

#save flat_chess to tmp file
cv2.imwrite('tmp/tmp33.jpg', flat_chess)

dst = cv2.cornerHarris(src=gray_real_chess, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

real_chess[dst>0.01*dst.max()] = [255,0,0]

#save real_chess to tmp file
cv2.imwrite('tmp/tmp34.jpg', real_chess)

#apply shi-tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray_flat_chess, 5, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x,y), 3, 255, -1)

#save flat_chess to tmp file
cv2.imwrite('tmp/tmp35.jpg', flat_chess)

corners = cv2.goodFeaturesToTrack(gray_real_chess, 5, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(real_chess, (x,y), 3, 255, -1)

#save real_chess to tmp file
cv2.imwrite('tmp/tmp36.jpg', real_chess)