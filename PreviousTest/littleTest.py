

import cv2 as cv
import numpy as np
import argparse
import os

# params for ShiTomasi corner detection

K = 300
maxCorners = max([K, 1]);
qualityLevel = 0.1
minDistance = 25
blockSize = 3
gradientSize = 3
useHarrisDetector = False
k = 0.04

feature_params = dict( maxCorners = maxCorners,
                       qualityLevel = qualityLevel,
                       minDistance = minDistance,
                       blockSize = blockSize,
                       gradientSize=gradientSize,
                       useHarrisDetector=useHarrisDetector,
                       k=k)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


# Take first frame and find corners in it
PATH = "../DAVIS/JPEGImages/480p/kite-walk/"

elements = os.listdir(PATH)

# sort frames
elements=sorted(elements)


# ret, old_frame = cap.read()
old_frame=cv.imread(PATH+elements[0])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

kernel = np.ones((3, 3), np.float32) / 9

# old_gray = cv.filter2D(old_gray, -1, kernel)

# Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

bckrmv1 = cv.createBackgroundSubtractorKNN()
bckrmv2 = cv.createBackgroundSubtractorMOG2()

for i, names in enumerate(elements):

    if i==0:
        continue

    # old_gray = cv.filter2D(old_gray, -1, kernel)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # ret,frame = cap.read()
    frame=cv.imread(PATH+names)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # calculate optical flow

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    H,_=cv.findHomography(good_old,good_new,cv.RANSAC)
    # print(H)
    height, width, channels = frame.shape
    img_est = cv.warpPerspective(old_gray,H,(width,height))
    img=cv.absdiff(frame_gray,img_est)
    # H = cv::findHomography( corners_prev, corners, cv::RANSAC );

    """
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    """

    kernel_a = np.ones((3,3))
    kernel_b = np.ones((5,5))

    # img = np.ones((img.shape))*(img > 10)
    img = cv.dilate(img, kernel_a, iterations=2)
    #
    # cv.imshow('frameb', img)
    #
    img = cv.erode(img, kernel_b, iterations=2)
    img = cv.dilate(img, kernel_a, iterations=2)

    # kernel_a = np.ones((2, 2))
    # img = cv.erode(img, kernel_a, iterations=1)
    #
    # cv.imshow('framee', img)
    #
    # kernel_a = np.array([[1,1,1],[1,1,1],[1,1,1]])
    # img = cv.dilate(img, kernel_a, iterations=1)

    cv.imshow('frame',img)
    # img=cv.medianBlur(img,3)
    # img = cv.edgePreservingFilter(img)*3

    fgMask = bckrmv1.apply(frame)
    fgMask = cv.bitwise_and(frame,frame,mask = fgMask)
    img=np.ones((img.shape),dtype=np.uint8)*(img>20)
    fgMask = cv.bitwise_and(fgMask,fgMask,mask = img )


    #
    # fgMask = cv.dilate(fgMask, kernel_a, iterations=2)
    #
    # fgMask = cv.erode(fgMask, kernel_a,iterations=4)
    #
    #
    # fgMask=cv.medianBlur(fgMask,3)
    # fgMask=cv.Canny(fgMask,400,600)
    # fgMask=cv.borderInterpolate()
    # fgMask = fgMask * (fgMask > 20)
    cv.imshow('frame2',fgMask)
    k = cv.waitKey() & 0xff
    if k == 27:
         break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)





"""
cv::goodFeaturesToTrack( prev_img_,
corners_prev,
maxCorners,
qualityLevel,
minDistance,
cv::Mat(),
blockSize,
/*gradientSize,*/
useHarrisDetector,
k );

/// Local motion estimation (DOUBT: is it bi-directional?)
cv::calcOpticalFlowPyrLK(prev_img_, img, corners_prev, corners, status, err, cv::Size(15,15), 2, criteria);
/// Global motion estimation

H = cv::Mat::zeros( img.rows, img.cols, img.type() );
// Get the Perspective Transform Matrix i.e. H
H = cv::findHomography( corners_prev, corners, cv::RANSAC );

// Compute background substracted image

cv::warpPerspective(prev_img_, img_estimated, H, img_estimated.size());
cv::absdiff(img, img_estimated, E_img);
"""
