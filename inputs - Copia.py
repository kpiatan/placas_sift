import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

MIN_MATCH_COUNT = 10

img1 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base01.jpg',0)         
img2 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base02.jpg',0)
img3 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base03.jpg',0)
img4 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base04.jpg',0)
img5 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base05.jpg',0)
img6 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base06.jpg',0)
img7 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base07.jpg',0)
img8 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base08.jpg',0)
img9 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base09.jpg',0)
img10 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base10.jpg',0)
img11 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base11.jpg',0)
img12 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base12.jpg',0)
img13 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base13.jpg',0)
img14 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base14.jpg',0)
img15 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base15.jpg',0)

imgtest = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_dia2/014.jpg',0)


#keypoints e descritores SIFT
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)
kp4, des4 = sift.detectAndCompute(img4,None)
kp5, des5 = sift.detectAndCompute(img5,None)
kp6, des6 = sift.detectAndCompute(img6,None)
kp7, des7 = sift.detectAndCompute(img7,None)
kp8, des8 = sift.detectAndCompute(img8,None)
kp9, des9 = sift.detectAndCompute(img9,None)
kp10, des10 = sift.detectAndCompute(img10,None)
kp11, des11 = sift.detectAndCompute(img11,None)
kp12, des12 = sift.detectAndCompute(img12,None)
kp13, des13 = sift.detectAndCompute(img13,None)
kp14, des14 = sift.detectAndCompute(img14,None)
kp15, des15 = sift.detectAndCompute(img15,None)

kptest, destest = sift.detectAndCompute(imgtest,None)

# parametros FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,destest,k=2)
matches2 = flann.knnMatch(des1,des2,k=2)
matches3 = flann.knnMatch(des1,des3,k=2)
matches4 = flann.knnMatch(des1,des4,k=2)
matches5 = flann.knnMatch(des1,des5,k=2)
matches6 = flann.knnMatch(des1,des6,k=2)
matches7 = flann.knnMatch(des1,des7,k=2)
matches8 = flann.knnMatch(des1,des8,k=2)
matches9 = flann.knnMatch(des1,des9,k=2)
matches10 = flann.knnMatch(des1,des10,k=2)
matches11 = flann.knnMatch(des1,des11,k=2)
matches12 = flann.knnMatch(des1,des12,k=2)
matches13 = flann.knnMatch(des1,des13,k=2)
matches14 = flann.knnMatch(des1,des14,k=2)
matches15 = flann.knnMatch(des1,des15,k=2)

# store all the good matches as per Lowe's ratio test.
good1 = []
for m,n in matches:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good1.append(m)

good2 = []
for m,n in matches2:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good2.append(m)

good3 = []
for m,n in matches3:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good3.append(m)

good4 = []
for m,n in matches4:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good4.append(m)

good5 = []
for m,n in matches5:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good5.append(m)

good6 = []
for m,n in matches6:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good6.append(m)

good7 = []
for m,n in matches7:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good7.append(m)

good8 = []
for m,n in matches8:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good8.append(m)

good9 = []
for m,n in matches9:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good9.append(m)

good10 = []
for m,n in matches10:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good10.append(m)

good11 = []
for m,n in matches11:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good11.append(m)

good12 = []
for m,n in matches12:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good12.append(m)

good13 = []
for m,n in matches13:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good13.append(m)

good14 = []
for m,n in matches14:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good14.append(m)

good15 = []
for m,n in matches15:
    if m.distance < 0.7*n.distance: # padrao 0.7
        good15.append(m)

good = []
a1 = []
a2 = []
a3 = []
a4 = []
a5 = []
a6 = []
a7 = []
a8 = []
a9 = []
a10 = []
a11 = []
a12 = []
a13 = []
a14 = []
a15 = []
for item in good1:
    a1.append(item.queryIdx)
for item in good2:
    a2.append(item.queryIdx)
for item in good3:
    a3.append(item.queryIdx)
for item in good4:
    a4.append(item.queryIdx)
for item in good5:
    a5.append(item.queryIdx)
for item in good6:
    a6.append(item.queryIdx)
for item in good7:
    a7.append(item.queryIdx)
for item in good8:
    a8.append(item.queryIdx)
for item in good9:
    a9.append(item.queryIdx)
for item in good10:
    a10.append(item.queryIdx)
for item in good11:
    a11.append(item.queryIdx)
for item in good12:
    a12.append(item.queryIdx)
for item in good13:
    a13.append(item.queryIdx)
for item in good14:
    a14.append(item.queryIdx)
for item in good15:
    a15.append(item.queryIdx)

#goodlist = good1
goodlist = list(set(a1).intersection(a2))
goodlist = list(set(goodlist).intersection(a3))
goodlist = list(set(goodlist).intersection(a4))
goodlist = list(set(goodlist).intersection(a5))
goodlist = list(set(goodlist).intersection(a6))
goodlist = list(set(goodlist).intersection(a7))
goodlist = list(set(goodlist).intersection(a8))
goodlist = list(set(goodlist).intersection(a9))
goodlist = list(set(goodlist).intersection(a10))
goodlist = list(set(goodlist).intersection(a11))
goodlist = list(set(goodlist).intersection(a12))
goodlist = list(set(goodlist).intersection(a13))
goodlist = list(set(goodlist).intersection(a14))
goodlist = list(set(goodlist).intersection(a15))

good = []
for item in goodlist:
    good.append(good1[goodlist.index(item)])

print(len(good))

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kptest[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    imgtest = cv.polylines(imgtest,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # casamento em verde
                singlePointColor = None,
                matchesMask = matchesMask, # somente inliers
                flags = 2)

imgout = cv.drawMatches(img1,kp1,imgtest,kptest,good,None,**draw_params)
#plt.imshow(imgout, 'gray'),plt.show()
cv.imwrite('f:/Downloads/VisaoComputacional/SiftRansac/placas_multiplosinputs/img014_15inp_.jpg',imgout)