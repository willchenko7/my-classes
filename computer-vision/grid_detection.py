import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('DATA/flat_chessboard.png')

#save flat_chess to tmp file
cv2.imwrite('tmp/tmp41.jpg', flat_chess)

found,corners = cv2.findChessboardCorners(flat_chess, (7,7))

cv2.drawChessboardCorners(flat_chess, (7,7), corners, found)

#save flat_chess to tmp file
cv2.imwrite('tmp/tmp42.jpg', flat_chess)


dots = cv2.imread('DATA/dot_grid.png')
found,corners = cv2.findCirclesGrid(dots, (10,10), cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(dots, (10,10), corners, found)

#save dots to tmp file
cv2.imwrite('tmp/tmp43.jpg', dots)
