import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import pickle

MIN_MATCH_COUNT = 10

path = "f:/Downloads/VisaoComputacional/SiftRansac/placas_base/"
pathout = "f:/Downloads/VisaoComputacional/SiftRansac/placas_out2_mult/"
dirs = os.listdir( path )
pathtest = 'f:/Downloads/VisaoComputacional/SiftRansac/placas_dia2/001.jpg'

img1 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base01.jpg',0)
imgtest = cv.imread(pathtest,0)
base=os.path.basename(pathtest)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kptest, destest = sift.detectAndCompute(imgtest,None)

# parametros FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

descriptlist = []
goodlist = []

for item in dirs:
        if os.path.isfile(path+item):
            img = cv.imread(path+item,0)
            kp, des = sift.detectAndCompute(img,None)
            matches = flann.knnMatch(des,destest,k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance: # padrao 0.7
                    good.append(m)
            goodlist.append(good)
            
            good_idx = []
            for item in good:
                good_idx.append(item.queryIdx)
            
            descriptlist.append(good_idx)

interdescriptlist = []
for i in range(len(descriptlist)-2):
    interdescript = [] # queryIdx comuns a 3 exemplos, 0 1 2, depois 0 1 3, etc
    for j in range(len(descriptlist)-2):
        for k in range(len(descriptlist)-2):
            interdescript.extend(set(descriptlist[i]).intersection(descriptlist[j+1],descriptlist[k+2]))
    interdescript = list(dict.fromkeys(interdescript)) # remove duplicatas
    interdescriptlist.append(interdescript) # matriz com todas as interdescripts

querysetinter = list(set(descriptlist[0]).intersection(interdescriptlist[0]))
for i in range(len(interdescriptlist)):
    querysetinter.extend(list(set(descriptlist[0]).intersection(interdescriptlist[i])))

querysetinter = list(dict.fromkeys(querysetinter))
good = []
for item in querysetinter:
    good.append(goodlist[0][querysetinter.index(item)])

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kptest[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    imgtest = cv.polylines(imgtest,[np.int32(dst)],True,255,3, cv.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # casamento em verde
                singlePointColor = None,
                matchesMask = matchesMask, # somente inliers
                flags = 2)

imgout = cv.drawMatches(img1,kp1,imgtest,kptest,good,None,**draw_params)
#plt.imshow(imgout, 'gray'),plt.show()
cv.imwrite(pathout+base,imgout)