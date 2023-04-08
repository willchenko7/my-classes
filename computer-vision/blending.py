import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read img1 and img2
img1 = cv2.imread('DATA/dog_backpack.png')
img2 = cv2.imread('DATA/watermark_no_copy.png')

#conver img1 and img2 to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#save img1 and img2 to tmp files
cv2.imwrite('tmp1.jpg', img1)
cv2.imwrite('tmp2.jpg', img2)

#resize imgs to be the same size
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

#blend img1 and img2
blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)

#save blended to tmp file
cv2.imwrite('tmp3.jpg', blended)

#overlay small img on top of a larger img (no blending)
img1 = cv2.imread('DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))

large_img = img1
small_img = img2

x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end, x_offset:x_end] = small_img

#save large_img to tmp file
cv2.imwrite('tmp4.jpg', large_img)

#overlay small img on top of a larger img (blending) with mask
img1 = cv2.imread('DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))

x_offset = 934 - 600
y_offset = 1401 - 600

rows, cols, channels = img2.shape
roi = img1[y_offset:1401, x_offset:934]

# create a mask of logo and create its inverse mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask_inv = cv2.bitwise_not(img2gray)
cv2.imwrite('tmp5.jpg', mask_inv)

white_background = np.full(img2.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
cv2.imwrite('tmp6.jpg', fg)
fg = cv2.bitwise_or(img2, img2, mask=mask_inv)

final_roi = cv2.bitwise_or(roi, fg)

img1[y_offset:1401, x_offset:934] = final_roi

#save img1 to tmp file
cv2.imwrite('tmp7.jpg', img1)