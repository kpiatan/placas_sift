import numpy as np
import cv2 as cv
img = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas/001.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)

#img=cv.drawKeypoints(gray,kp,img)
#cv.imwrite('f:/Downloads/VisaoComputacional/SiftRansac/out/cd1.jpg',img)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('f:/Downloads/VisaoComputacional/SiftRansac/out/placa1.jpg',img)