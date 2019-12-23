import numpy as np
import cv2

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
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class HomographyFilter:

    def __init__(self):

        self.old_gray = None

    def __call__(self,frame):

        if self.old_gray is None:
            print("oldGray is none")
            self.old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            return self.old_gray

        p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **feature_params)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC)
        # print(H)
        height, width, channels = frame.shape
        img_est = cv2.warpPerspective(self.old_gray, H, (width, height))
        img = cv2.absdiff(frame_gray, img_est)

        self.old_gray = frame_gray.copy()
        # self.p0 = good_new.reshape(-1, 1, 2)


        return img






