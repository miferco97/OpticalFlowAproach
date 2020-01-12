import numpy as np
import cv2


class FrameComparison:

    def __init__(self):
        self.prev_gray = None

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Convertir a escala de grises

        if self.prev_gray is None:
            self.prev_gray = gray

        difference = cv2.absdiff(self.prev_gray, gray)  # Resta absoluta
        _, thres = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)  # Aplicar umbral
        #     thres2 = cv2.dilate(thres, None, iterations=2)

        ###
        frame, contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Buscar contornos
        for c in contours:
            # Eliminamos los contornos mas pequenos
            if cv2.contourArea(c) < 500:
                continue

            # Obtenemos el bounds del contorno, el rectangulo mayor que engloba al contorno
            (x, y, w, h) = cv2.boundingRect(c)
            # Dibujamos el rectangulo del bounds
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ###
        # cv2.imshow("Prev frame", thres)
        # cv2.imshow("Frame", frame)
        # cv2.imshow("Difference", difference)

        self.prev_gray = gray

        # Mostramos las capturas
        # cv2.imshow('Camara', frame)
        # cv2.imshow('Umbral', fgmask)
        # cv2.imshow('Contornos', contornosimg)

        # Sentencias para salir, pulsa 's' y sale
        k = cv2.waitKey(1) & 0xff

        return gray