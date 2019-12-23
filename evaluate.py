from HomographyFilter import *
import cv2
import os

# Take first frame and find corners in it
PATH = "./DAVIS/JPEGImages/480p/kite-walk/"

elements = os.listdir(PATH)

# sort frames
elements=sorted(elements)

filter1 = HomographyFilter()

for i, names in enumerate(elements):

    frame=cv2.imread(PATH+names)
    resultMask = filter1(frame)
    cv2.imshow('frame1',resultMask)
    k = cv2.waitKey() & 0xff
    if k == 27:
         break


