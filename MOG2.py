import numpy as np
import cv2


class MOG2:

    def __init__(self):
        # Llamada al metodo
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        # Deshabilitamos OpenCL, si no hacemos esto no funciona
        cv2.ocl.setUseOpenCL(False)

    def __call__(self, frame):

        # Aplicamos el algoritmo
        fgmask = self.fgbg.apply(frame)

        # Copiamos el umbral para detectar los contornos
        contornosimg = fgmask.copy()

        # Buscamos contorno en la imagen
        frame, contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Recorremos todos los contornos encontrados
        for c in contornos:
            # Eliminamos los contornos mas pequenos
            if cv2.contourArea(c) < 500:
                continue

            # Obtenemos el bounds del contorno, el rectangulo mayor que engloba al contorno
            (x, y, w, h) = cv2.boundingRect(c)
            # Dibujamos el rectangulo del bounds
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostramos las capturas
        # cv2.imshow('Camara', frame)
        # cv2.imshow('Umbral', fgmask)
        # cv2.imshow('Contornos', contornosimg)

        # Sentencias para salir, pulsa 's' y sale
        k = cv2.waitKey(1) & 0xff

        return fgmask