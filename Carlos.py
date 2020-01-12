import numpy as np
import cv2 as cv
import os
from scipy.signal import find_peaks
from scipy import stats
from matplotlib import pyplot as plt



class DenseOpFlow:

    def __init__(self,frame):
        self.prvs_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        self.next_img = None
        self.hsv = np.zeros_like(frame)
        self.ones = np.ones([frame.shape[0],frame.shape[1]])
        self.aux = np.zeros_like(self.ones)
        self.hsv[...,1] = 255
        self.thr = 5  # 50
        self.th_bin = 10

    def __call__(self, frame, imageName):
        self.next_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.prvs_img,self.next_img, None, 0.5, 3, 20, 5, 7, 1.5, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

        # El color de la imagen se establece en funcion del angulo de movimiento de cada pixel
        self.hsv[..., 0] = cv.normalize(ang * 180 / np.pi / 2.0, None, 0, 255, cv.NORM_MINMAX)
        # print(self.hsv)

        # La intensidad de la imagen se establece en funcion de la magnitud del movimiento de cada pixel
        self.hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Se filtra la imagen
        kernel = np.ones((5, 5), np.float32) / 25
        self.hsv[...,2] = cv.filter2D(self.hsv[...,2], -1, kernel)

        # Se muestra la imagen filtrada
        bgr = cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR)
        cv.imshow('Ang + Mag - Filtrados', bgr)
        cv.imwrite("./Images/" + imageName, bgr)

        # Eliminar los pixels que tengan como angulo el valor mas repetido en la imagen (fondo)
        m = stats.mode(self.hsv[..., 0], axis=None)

        thr = (np.max(self.hsv[..., 0]) - np.min(self.hsv[..., 0])) / 2
        thr = 50


        hist = cv.calcHist([self.hsv], [0], None, [255], [0, 255])
        # plt.plot(hist, color='b')
        # plt.xlim([0, 255])
        # plt.show()
        # zeros = hist[0]

        # k = cv.waitKey() & 0xff

        # Se calcula el angulo de desplazamiento del fondo
        n_ang_fondo = np.max(hist)
        # Se elimina el fondo del histograma
        # hist[np.where(hist == ang_fondo)]
        peaks, _ = find_peaks(hist.T[0],height=10000, distance = 20)

        # print(peaks)

        # if(peaks.shape[0] >= 2):
            # dist = abs(peaks[1]-peaks[0])
            # if dist < 255/2:
            #     thr = dist/2
            # else:
            #     thr = (255 - dist)/2
            #
            # print("dist",dist)
            # print("thr",thr)

        self.aux[...] = abs(self.hsv[..., 0] - self.ones * m[0])
        if m[0] < thr or m[0] > 255 - thr:
            self.aux[self.aux > (255 - thr)] = 0
        self.aux[...] -= self.ones * thr
        self.aux[self.aux < 0] = 0

        self.aux[self.aux > self.th_bin] = 255

        # bckgF = cv.bitwise_and(self.hsv, self.hsv, mask=self.aux.astype('uint8'))
        # hist = cv.calcHist([bckgF], [0], None, [255], [0, 255])
        # hist[0] = n_ang_fondo
        # plt.plot(hist, color='b')
        # plt.xlim([0, 255])
        # bckgF = cv.cvtColor(bckgF, cv.COLOR_HSV2BGR)
        # cv.imshow('bckgF',bckgF)
        # plt.show()

        # Se muestra la imagen eliminando angulos
        ##cv.imshow('aux', self.aux)
        # k = cv.waitKey() & 0xff


        # for i in range(0,self.hsv.shape[0]):
        #     for j in range(0, self.hsv.shape[1]):
        #         if (self.hsv[i, j, 0] >= (m[0] - thr)) and (self.hsv[i, j, 0] <= (m[0] + thr)):
        #             self.hsv[i, j, 2] = 0

        # Se transforma la imagen a formato BGR
        bgr = cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR)

        # Se convierte a escala de grises
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        # Se binariza la imagen
        ret, th = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
        # th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        res = cv.bitwise_and(self.next_img, self.next_img, mask=self.aux.astype('uint8'))

        # cv.imshow('img', self.next_img)
        # cv.imshow('gray', gray)
        # cv.imshow('frame2', bgr)
        # cv.imshow('resultado', res)
        # k = cv.waitKey(30) & 0xff

        # print(self.hsv[:,:,0])
        # print(self.hsv[:,:,2])

        # if k == 27:
        #     break

        self.prvs_img = self.next_img

        return self.aux