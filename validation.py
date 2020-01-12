from HomographyFilter import *
from Carlos import *
from KNN import *
from MOG1 import *
from MOG2 import *
from FrameComparison import *
from tools import *
import cv2
import os

# filename = "train"
BASE_PATH = "./DAVIS"

all_filenames = os.listdir(BASE_PATH + "/JPEGImages/480p/")
all_filenames = sorted(all_filenames)

resultsFilename = "resultDOF_1.txt"
resultsPath = "./Resultados/" + resultsFilename

respuesta = 'respuesta'

respuesta = raw_input("Quieres realizar la validacion? s/n: ")
if respuesta == 's' or respuesta == 'S':
    if not os.path.isfile(resultsPath):
        f = open(resultsPath, "w")
    else:
        print("Ya existe un archivo con el nombre ", resultsFilename)
        while respuesta != 's' and respuesta != 'S' and respuesta != 'n' and respuesta != 'N' and respuesta != 'exit' and respuesta != 'EXIT' and respuesta!= 'none' and respuesta!='NONE':
            print("Desea sobreescribirlo?: s/n")
            respuesta = raw_input()
            if respuesta == 'exit' or respuesta == 'EXIT':
                exit()
            elif respuesta == 's' or respuesta == 'S':
                f = open(resultsPath, "w")
            elif respuesta == 'n' or respuesta == 'N':
                respuesta = raw_input("Escriba el nombre del nuevo archivo sin la extension: ")
                if respuesta == 'exit' or respuesta == 'EXIT':
                    exit()
                elif respuesta == 'none' or respuesta == 'NONE':
                    resultsFilename = respuesta
                else:
                    resultsFilename = respuesta
                    resultsPath = "./Resultados/" + resultsFilename
                    f = open(resultsPath, "w")

            else:
                print('Responda s/n/exit:')
else:
    resultsFilename = 'none'

all_filenames = ["bear"]

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
    filterDOF = DenseOpFlow(frame)
    filterKNN = KNN()
    filterMOG1 = MOG1()
    filterMOG2 = MOG2()
    filterFrameComparison = FrameComparison()

    n_img = 0

    precision = []
    recall = []
    F = []

    for i, names in enumerate(elements):
        frame = cv2.imread(PATH+names)
        annotation = cv2.imread(ANNOTATIONS_PATH + annotations_elements[n_img], 0)
        n_img += 1

        #----- HOMOGRAPHY FLITER -----#

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

        # cv2.imshow('frame3', finalMask)

        #----- DENSE OPTICAL FLOW -----#

        resultMask_Carl = filterDOF(frame, filename + "_" + names)

        res = cv2.bitwise_and(frame, frame, mask=resultMask_Carl.astype('uint8'))

        cv2.imshow('image', frame)
        cv2.imshow('mask', resultMask_Carl)
        cv2.imshow('result', res)
        cv2.imshow('annotation', annotation)

        cv2.imwrite("./Mascaras/" + filename + "_" + names, resultMask_Carl)
        cv2.imwrite("./MaskResults/" + filename + "_" + names, res)

        #----- KNN -----#

        # resultKNN = filterKNN(frame)
        #
        # res = cv2.bitwise_and(frame, frame, mask=resultKNN.astype('uint8'))
        #
        # cv2.imshow('image', frame)
        # cv2.imshow('mask', resultKNN)
        # cv2.imshow('result', res)
        # cv2.imshow('annotation', annotation)

        # ----- MOG1 -----#

        # resultMOG1 = filterMOG1(frame)
        #
        # res = cv2.bitwise_and(frame, frame, mask=resultMOG1.astype('uint8'))
        #
        # cv2.imshow('image', frame)
        # cv2.imshow('mask', resultMOG1)
        # cv2.imshow('result', res)
        # cv2.imshow('annotation', annotation)

        # ----- MOG2 -----#

        # resultMOG2 = filterMOG2(frame)
        #
        # res = cv2.bitwise_and(frame, frame, mask=resultMOG2.astype('uint8'))
        #
        # cv2.imshow('image', frame)
        # cv2.imshow('mask', resultMOG2)
        # cv2.imshow('result', res)
        # cv2.imshow('annotation', annotation)

        # ----- FrameComparison -----#

        # resultFrameComparison = filterFrameComparison(frame)
        #
        # res = cv2.bitwise_and(frame, frame, mask=resultFrameComparison.astype('uint8'))
        #
        # cv2.imshow('image', frame)
        # cv2.imshow('mask', resultFrameComparison)
        # cv2.imshow('result', res)
        # cv2.imshow('annotation', annotation)


        # Calculate validation parameters

        resultForValidation = resultMask_Carl

        comp = cv2.bitwise_and(resultForValidation, resultForValidation, mask=annotation.astype('uint8'))

        tp = len(comp[comp > 0])
        TotalPositive = len(resultForValidation[resultForValidation > 0])
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


    # Calculate Validation Parameters for the Video
    TotalPrecision.append(np.mean(precision))
    TotalRecall.append(np.mean(recall))
    TotalF.append(2*TotalRecall[filenumber]*TotalPrecision[filenumber]/(TotalRecall[filenumber]+TotalPrecision[filenumber]))

    # print('Moda de la precision', stats.mode(precision, axis=None))
    print(filename)
    print('Media de la precision', TotalPrecision[filenumber])
    print('Media de la exhaustividad', TotalRecall[filenumber])
    print('F', TotalF[filenumber])
    print("\n")

    filenumber += 1

if resultsFilename != 'none' and resultsFilename != 'NONE':
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




