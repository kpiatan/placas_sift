from __future__ import print_function
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

MIN_MATCH_COUNT = 10

img1 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base01.jpg', 0)
img2 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_dia2/001.jpg', 0)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# parametros FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
  if m.distance < 0.5*n.distance:  # padrao 0.7
    good.append(m)

if len(good) > MIN_MATCH_COUNT:
  src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
  dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
  M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
  matchesMask = mask.ravel().tolist()
  h, w = img1.shape
  pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1,1,2)
  dst = cv.perspectiveTransform(pts, M)
  #img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
  p1 = np.int32(dst[0][0][0])
  p2 = np.int32(dst[0][0][1])
  p3 = np.int32(dst[2][0][0])
  p4 = np.int32(dst[2][0][1])
  pre = np.array([[1, 0, -p1], 
    [0, 1, -p2],
    [0, 0, 1]])
  M = np.matmul(pre,np.array(M))

# Use homography
#img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None)
#plt.imshow(img3, 'gray'), plt.show()
img4 = cv.warpPerspective(img2, M, (w, h))
plt.imshow(img4, 'gray'), plt.show()