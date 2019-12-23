from HomographyFilter import *
import os

# Take first frame and find corners in it
PATH = "./DAVIS/JPEGImages/480p/kite-walk/"

elements = os.listdir(PATH)

# sort frames
elements=sorted(elements)

filter1 = HomographyFilter()

for i, names in enumerate(elements):

    frame=cv.imread(PATH+names)
    resultMask = filter(frame)
    cv.imshow('frame1',resultMask)
    k = cv.waitKey() & 0xff
    if k == 27:
         break


