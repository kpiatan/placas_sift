import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

MIN_MATCH_COUNT = 10
SKIP_MAX = 30 #30 
olddst = np.zeros((4,1,2))
framestreak = 0
framebonus = 2
frametotals = 0
framecorrect = 0

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv.VideoCapture('f:/Downloads/VisaoComputacional/SiftRansac/placas_vid/02.mp4')
#base=os.path.basename('/root/dir/sub/file.ext')
#os.path.splitext(base)[0]
img1 = cv.imread('f:/Downloads/VisaoComputacional/SiftRansac/placas_base/base01.jpg',0)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file. 
out = cv.VideoWriter('f:/Downloads/VisaoComputacional/SiftRansac/vid_out/teste.avi',cv.VideoWriter_fourcc(*'DIVX'), 20, (frame_width,frame_height))

# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img2 = frame
        frametotals = frametotals+1

        # keypoints e descritores SIFT
        kp2, des2 = sift.detectAndCompute(img2,None)

        # parametros FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance: # padrao 0.7
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
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
                framecorrect = framecorrect+1
            elif (framebonus < 3 and framebonus > 0):
                framebonus = framebonus-1
            else:    
                framestreak = 0

            if framestreak > 4:
                img2 = cv.polylines(img2,[np.int32(dst)],True,(0, 255, 0),3, cv.LINE_AA)
            else: 
                img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
            

        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # casamento em verde
                        singlePointColor = None,
                        matchesMask = matchesMask, # somente inliers
                        flags = 2)

        img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        #img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)

        font = cv.FONT_HERSHEY_SIMPLEX

        videostr = 'Frames: ' + str(framestreak)
  
        # Use putText() method for 
        # inserting text on video
        if framestreak > 4:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv.putText(img2,  
                    videostr,  
                    (50, 50),  
                    font, 1,  
                    color,  # blue, green, red
                    3,  
                    cv.LINE_4) 

        # write the flipped frame
        out.write(img2)
        
        cv.imshow('video',img2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv.destroyAllWindows()
print(frametotals)
print(framecorrect)