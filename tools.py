import cv2

def umbralize(frame,threshold = 127):
    ret, frame_umbralized = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    return frame_umbralized

def erode_and_dilate(frame,structuringElementSize = 3):
    erosion_type = cv2.MORPH_RECT
    element = cv2.getStructuringElement(erosion_type, (structuringElementSize,structuringElementSize))
    frame = cv2.erode(frame,element,iterations=2)
    frame = cv2.dilate(frame, element, iterations=1)

    return frame

def dilate_and_erode(frame,structuringElementSize = 3):
    erosion_type = cv2.MORPH_RECT
    element = cv2.getStructuringElement(erosion_type, (structuringElementSize,structuringElementSize))
    frame = cv2.dilate(frame, element, iterations=1)
    frame = cv2.erode(frame, element, iterations=1)
    # frame = cv2.dilate(frame, element, iterations=1)

    return frame


class BgSubstractor:

    def __init__(self,bgSubstractor="MOG2"):
        if bgSubstractor == "MOG2":
            self.bgSubs = cv2.createBackgroundSubtractorMOG2()
        elif bgSubstractor == "KNN":
            self.bgSubs = cv2.createBackgroundSubtractorKNN()
        else:
            raise Exception("No correct Backgroung Substractor selected")

    def __call__(self,frame):
        return self.bgSubs.apply(frame)
