import cv2
import numpy as np
import matplotlib.pyplot as plt


reeses = cv2.imread('DATA/reeses_puffs.png',0)

#save img to tmp file
cv2.imwrite('tmp/tmp47.jpg', reeses)

cereals = cv2.imread('DATA/many_cereals.jpg',0)

#save img to tmp file
cv2.imwrite('tmp/tmp48.jpg', cereals)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x:x.distance)

reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)

#save img to tmp file
cv2.imwrite('tmp/tmp49.jpg', reeses_matches)

#use sift
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

sift_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=2)

#save img to tmp file
cv2.imwrite('tmp/tmp50.jpg', sift_matches)

#use flann
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=0)

#save img to tmp file
cv2.imwrite('tmp/tmp51.jpg', flann_matches)