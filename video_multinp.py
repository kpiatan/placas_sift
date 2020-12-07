import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

MIN_MATCH_COUNT = 10
SKIP_MAX = 30 #30 
olddst = np.zeros((4,1,2))
framestreak = 0
framebonus = 2

path = "f:/Downloads/VisaoComputacional/SiftRansac/placas_base/"
dirs = os.listdir( path )
cap = cv.VideoCapture('f:/Downloads/VisaoComputacional/SiftRansac/placas_vid/05.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file. 
out = cv.VideoWriter('f:/Downloads/VisaoComputacional/SiftRansac/vid_out_mult/05.avi',cv.VideoWriter_fourcc(*'DIVX'), 20, (frame_width,frame_height))

img1 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base01.jpg',0)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        imgtest = frame        
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
            if 'dst' in locals():
                olddst = dst
            dst = cv.perspectiveTransform(pts,M)
            if 'dst' not in locals():
                olddst = dst

            if ((olddst[0][0][0]-dst[0][0][0] < SKIP_MAX) and #situacao ideal
                (olddst[1][0][0]-dst[1][0][0] < SKIP_MAX) and
                (olddst[2][0][0]-dst[2][0][0] < SKIP_MAX) and
                (olddst[3][0][0]-dst[3][0][0] < SKIP_MAX) and
                (olddst[0][0][0]-dst[0][0][0] > -SKIP_MAX ) and 
                (olddst[1][0][0]-dst[1][0][0] > -SKIP_MAX ) and
                (olddst[2][0][0]-dst[2][0][0] > -SKIP_MAX ) and
                (olddst[3][0][0]-dst[3][0][0] > -SKIP_MAX )):
                framestreak = framestreak+1
                framebonus = 2
            elif (framebonus < 3 and framebonus > 0):
                framebonus = framebonus-1
            else:    
                framestreak = 0

            if framestreak > 4:
                imgtest = cv.polylines(imgtest,[np.int32(dst)],True,(0, 255, 0),3, cv.LINE_AA)
            else: 
                imgtest = cv.polylines(imgtest,[np.int32(dst)],True,255,3, cv.LINE_AA)
 
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        font = cv.FONT_HERSHEY_SIMPLEX

        videostr = 'Frames: ' + str(framestreak)
  
        # Use putText() method for 
        # inserting text on video
        if framestreak > 4:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv.putText(imgtest,  
                    videostr,  
                    (50, 50),  
                    font, 1,  
                    color,  # blue, green, red
                    3,  
                    cv.LINE_4) 

        # write the flipped frame
        out.write(imgtest)        
        cv.imshow('video',imgtest)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv.destroyAllWindows()