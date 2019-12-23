import cv2,os
import numpy as np

# median = cv2.medianBlur(img,5)

def ComparativeBackgroundSubstract(frame0,frame1,threshold = 30):
    # kernel = np.ones((3, 3), np.float32) / 9
    # backSub = cv2.createBackgroundSubtractorMOG2()

    mask0 = backSub.apply(frame0)

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame0 = cv2.bitwise_and(frame0, frame0, mask=mask0)

    # frame0 = cv2.filter2D(frame0, -1, kernel)
    # frame0 = cv2.medianBlur(frame0, 5)
    mask1 = backSub.apply(frame1)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # frame1 = cv2.medianBlur(frame1, 5)
    frame1 = cv2.bitwise_and(frame1,frame1, mask=mask1)

    # frame1 = cv2.filter2D(frame1, -1, kernel)

    subs = np.abs(frame0.astype(int)- frame1.astype(int)).astype(np.uint8)
    subs = cv2.medianBlur(subs, 5)

    # print(frame0[0][0],frame1[0][0],subs[0][0])


    # subs = cv2.filter2D(subs, -1, kernel)

    _, mask = cv2.threshold(subs, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    result_im = cv2.bitwise_and(subs,mask)
    return result_im

PATH = "../DAVIS/JPEGImages/480p/bus/"

elements = os.listdir(PATH)
# sort frames
elements=sorted(elements)

for k in range(len(elements)):

    kernel = np.ones((9,9), np.float32) / 81

    frame0 = cv2.imread(PATH+elements[k+0])
    frame1 = cv2.imread(PATH+elements[k+1])
    frame0 = cv2.filter2D(frame0, -1, kernel)
    frame0 = cv2.filter2D(frame0, -1, kernel)
    frame0 = cv2.filter2D(frame0, -1, kernel)
    frame0 = cv2.filter2D(frame0, -1, kernel)
    # backSub = cv2.createBackgroundSubtractorKNN()

    # mask0 = backSub.apply(frame0)

    # print(fra0)
    # frame0 = cv2.bitwise_and(frame0, frame0, mask=mask0)

    cv2.imshow("res2", frame0)

    img=ComparativeBackgroundSubstract(frame0,frame1)

    # im = img.copy()

    area = []
    goodContours = []
    edged = cv2.Canny(img, 30, 200)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))

    for i in range(len(contours)):
        if contours[i] is not None:
            area_ = cv2.contourArea(contours[i])
            print(area_)
            if area_ > 200 and area_ <2000:
                goodContours += [contours[i]]
                area += [area_]

    im=np.zeros(img.shape)
    im2=im.copy()
    cv2.drawContours(im, goodContours, -1, (255, 255, 255), 3)
    cv2.drawContours(im2, contours, -1, (255, 255, 255), 3)

    # cv2.drawContours(im, contours, -1, (0,255,0), 3)

    cv2.imshow("bestContours", im)
    cv2.imshow("contours", im2)

    cv2.waitKey()