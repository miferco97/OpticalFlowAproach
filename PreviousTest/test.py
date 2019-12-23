import cv2 as cv
import cv2
import os
import numpy as np


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv2.createBackgroundSubtractorKNN()
PATH = "../DAVIS/JPEGImages/480p/kite-walk/"


elements=sorted(os.listdir(PATH))

old_frame = cv2.imread(PATH + elements[0])

filter_strenght = 30

for i in range(len(elements)):
    frame = cv2.imread(PATH + elements[i])

    f = old_frame - frame
    f_ = (f < 120).astype(np.uint8)
    f *= f_
    frame_gray=cv.cvtColor(f, cv.COLOR_BGR2GRAY)

    if i == 0:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()


    print(f)

    dst = cv2.fastNlMeansDenoisingColored(f, None, filter_strenght, filter_strenght, 7, 21)
    cv2.imshow('sub', dst)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    keyboard = cv2.waitKey(2)
    old_frame=frame
