from HomographyFilter import *
from Carlos import *
from tools import *
import cv2
import os

PATH = "./DAVIS/JPEGImages/480p/bus/"

elements = os.listdir(PATH)

frame = cv2.imread(PATH+elements[0])

# sort frames
elements = sorted(elements)
filter1 = HomographyFilter()
filter2 = BgSubstractor()
filter3 = DenseOpFlow(frame)

for i, names in enumerate(elements):
    frame = cv2.imread(PATH+names)

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

    resultMask_Carl = filter3(frame)

    res = cv2.bitwise_and(frame, frame, mask=resultMask_Carl.astype('uint8'))

    cv2.imshow('image', frame)
    cv2.imshow('mask', resultMask_Carl)
    cv2.imshow('result', res)

    k = cv2.waitKey() & 0xff
    if k == 27:
         break


