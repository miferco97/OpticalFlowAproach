from HomographyFilter import *
from tools import *
import cv2
import os

PATH = "./DAVIS/JPEGImages/480p/kite-walk/"

elements = os.listdir(PATH)
# sort frames
elements=sorted(elements)
filter1 = HomographyFilter()
filter2 = BgSubstractor()

for i, names in enumerate(elements):
    frame=cv2.imread(PATH+names)
    resultMask = filter1(frame)
    resultMask = umbralize(resultMask,threshold=30)
    resultMask = dilate_and_erode(resultMask,5)

    cv2.imshow('frame1', resultMask)
    bgSubstracted=filter2(frame)
    bgSubstracted = erode_and_dilate(bgSubstracted, 3)
    cv2.imshow('frame2', bgSubstracted)

    finalMask = bgSubstracted + resultMask
    finalMask= erode_and_dilate(finalMask,3)

    cv2.imshow('frame3', finalMask)

    k = cv2.waitKey() & 0xff
    if k == 27:
         break


