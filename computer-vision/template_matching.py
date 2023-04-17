import cv2
import numpy as np
import matplotlib.pyplot as plt

full = cv2.imread('DATA/sammy.jpg')
#full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

#save full to tmp file
cv2.imwrite('tmp/tmp28.jpg', full)

face = cv2.imread('DATA/sammy_face.jpg')

#save face to tmp file
cv2.imwrite('tmp/tmp29.jpg', face)


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    full_copy = full.copy()
    method = eval(m)
    res = cv2.matchTemplate(full_copy, face, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    height, width, channels = face.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(full_copy, top_left, bottom_right, (255,0,0), 10)
    #save full_copy to tmp file
    cv2.imwrite('tmp/tmp30.jpg', full_copy)