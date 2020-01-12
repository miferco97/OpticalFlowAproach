from HomographyFilter import *
from Carlos import *
from tools import *
import cv2
import os

# filename = "train"
BASE_PATH = "./DAVIS"

all_filenames = os.listdir(BASE_PATH + "/JPEGImages/480p/")
all_filenames = sorted(all_filenames)

f=open("resultados3.txt","w")

TotalPrecision = []
TotalRecall = []
TotalF = []

filenumber = 0

for k, filename in enumerate(all_filenames):

    PATH = BASE_PATH + "/JPEGImages/480p/" + filename + "/"
    ANNOTATIONS_PATH = BASE_PATH + "/Annotations/480p/" + filename + "/"

    elements = os.listdir(PATH)
    annotations_elements = os.listdir(ANNOTATIONS_PATH)

    frame = cv2.imread(PATH+elements[0])


    # sort frames
    elements = sorted(elements)
    annotations_elements = sorted(annotations_elements)

    filter1 = HomographyFilter()
    filter2 = BgSubstractor()
    filter3 = DenseOpFlow(frame)

    n_img = 0

    precision = []
    recall = []
    F = []

    for i, names in enumerate(elements):
        frame = cv2.imread(PATH+names)
        annotation = cv2.imread(ANNOTATIONS_PATH + annotations_elements[n_img], 0)
        n_img += 1

        # resultMask = filter1(frame)
        # resultMask = umbralize(resultMask,threshold=30)
        # resultMask = dilate_and_erode(resultMask,5)
        #
        # # cv2.imshow('frame1', resultMask)
        # bgSubstracted=filter2(frame)
        # bgSubstracted = erode_and_dilate(bgSubstracted, 3)
        # # cv2.imshow('frame2', bgSubstracted)
        #
        # finalMask = bgSubstracted + resultMask
        # finalMask= erode_and_dilate(finalMask,3)
        #
        # # cv2.imshow('frame3', finalMask)

        resultMask_Carl = filter3(frame)

        res = cv2.bitwise_and(frame, frame, mask=resultMask_Carl.astype('uint8'))

        # cv2.imshow('image', frame)
        cv2.imshow('mask', resultMask_Carl)
        # cv2.imshow('result', res)

        cv2.imshow('annotation', annotation)

        comp = cv2.bitwise_and(resultMask_Carl, resultMask_Carl, mask=annotation.astype('uint8'))

        cv2.imshow('validation', comp)

        # print(comp)

        # hist = cv.calcHist(annotation, [0], None, [256], [0, 256])
        # plt.plot(hist, color='b')
        # plt.xlim([0, 256])
        # plt.show()

        tp = len(comp[comp > 0])
        TotalPositive = len(resultMask_Carl[resultMask_Carl > 0])
        GroundTruthPositives = len(annotation[annotation > 0])
        if TotalPositive != 0:
            precision.append(1.0*tp/TotalPositive)
        else:
            precision.append(0)

        if GroundTruthPositives != 0:
            recall.append(1.0 * tp / GroundTruthPositives)
        else:
            recall.append(1.0)

        # print('Verdaderos Positivos', tp)
        # print('Verdaderos Positivos + Falsos Positivos', TotalPositive)
        # print('Verdaderos Positivos + Falsos Negativos', GroundTruthPositives)
        # print('Precision', precision[n_img-1])
        # print('Recall', recall[n_img - 1])

        k = cv2.waitKey(1) & 0xff
        if k == 27:
             break


    # print('Moda de la precision', stats.mode(precision, axis=None))
    print(filename)
    print('Media de la precision', np.mean(precision))
    print('Media de la exhaustividad', np.mean(recall))
    print("\n")

    TotalPrecision.append(np.mean(precision))
    TotalRecall.append(np.mean(recall))
    TotalF.append(2*TotalRecall[filenumber]*TotalPrecision[filenumber]/(TotalRecall[filenumber]+TotalPrecision[filenumber]))

    filenumber += 1


TotalF, all_filenames, TotalPrecision, TotalRecall = zip(*sorted(zip(TotalF, all_filenames, TotalPrecision, TotalRecall)))

filenumber = 0

for i in range(len(all_filenames)):
    f.write(all_filenames[i])
    f.write("\n")
    f.write('Media de la precision:  ')
    f.write(str(TotalPrecision[i]))
    f.write("\n")
    f.write('Media de la exhaustividad:  ')
    f.write(str(TotalRecall[i]))
    f.write("\n")
    f.write('F:  ')
    f.write(str(TotalF[i]))
    f.write("\n")
    f.write("\n")

    filenumber += 1


f.close()




