import cv2 as cv
import numpy as np
import os
from PreviousTest.utils import ComparativeBackgroundSubstract

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


# Take first frame and find corners in it
PATH = "../DAVIS/JPEGImages/480p/dog/"

elements = os.listdir(PATH)

# sort frames
elements=sorted(elements)

#backSub = cv.createBackgroundSubtractorMOG2()
backSub = cv.createBackgroundSubtractorKNN()

filter_strenght = 60
print(elements)


# ret, old_frame = cap.read()
old_frame=cv.imread(PATH+elements[0])
frame = cv.imread(PATH+elements[1])

old_gray=ComparativeBackgroundSubstract(old_frame,frame)

# old_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)


# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

elements=elements[2:]
for names in elements:
    # ret,frame = cap.read()
    frame=cv.imread(PATH+names)

    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_gray = ComparativeBackgroundSubstract(old_frame, frame)
    frame_ = frame_gray.copy()

    # calculate optical flow

    # p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #
    # # Select good points
    # good_new = p1[st==1]
    # good_old = p0[st==1]
    # # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new, good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     # mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame_ = cv.circle(frame_,(a,b),5,color[i].tolist(),-1)
    # # img = cv.add(frame_,mask)

    # canny = cv.Canny(frame_gray,3,5)

    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    area = []
    goodContours = []
    try:
        for i in range(len(contours)):
            area = area + [cv.contourArea(contours[i])]
            # print(area[i])
            if area[i] > 1000:
                goodContours += [contours[i]]
            im2 = cv.drawContours(frame_gray, goodContours, 5, (0, 255, 0), 3)
            cv.imshow('im2', im2)

    except :
        a=1

    # cv.imshow('canny', canny)

    cv.imshow('frame',frame_)
    # cv.imshow('frame2',fgMask)
    k = cv.waitKey() & 0xff
    if k == 27:
         break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    old_frame =frame.copy()
    # p0 = good_new.reshape(-1,1,2)
