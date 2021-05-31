"""
This module make HOG

Используем скользящие окна и HOG (гистограммы направленных градиентов) на изображении для обнаружения лиц

Направленные градиенты

Author Gansior A. mail - gansior@gansior.ru tel - +79173383804
"""

import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt
from skimage import exposure,feature
import face_recognition

def show_image (i):
    plt.figure(figsize=(16,8))
    plt.imshow(i[...,::-1])
    plt.show()

if __name__ == '__main__':
    image = cv2.imread('faces.example.png')
    # HOG
    #
    # Параметры:
    #
    #     winSize - размер характерного объекта на изображении, который должен описываться картой градиентов. Типичное лицо находится в квадрате 20-30 пикселей, 32 кратно 2
    #     blockSize - размер блоков, на которые будем разбивать характерное изображение, кратно 2.
    #     blockStride - смещение блоков для анализа. Выбирают либо 50%, либо 25% от размера блока.
    #     cellSize - размер ячейки для нормализации яркости, обычно ставят равным размеру блока изображения или в 2 раза меньше.
    #     nbins - число ячеек (столбцов) в гистограмме. Поскольку выбираем ориентацию по плоскости (360 градусов), имеет смысл выставить в 32.
    #     signedGradients - ориентация по плоскости (360 градусов) или полуплоскости (180 градусов). Больше - лучше.

    winSize = (32, 32)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 32
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                            cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels, useSignedGradients)

    fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=blockSize,
                                cells_per_block=(2, 2), transform_sqrt=True,
                                block_norm="L1", visualize=True)
    hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
    show_image(hog_image)

    # Загрузка обученных моделей
    #
    # Пакет face_recognition
    face_locations = face_recognition.face_locations(image)
    img = image.copy()
    for top, right, bottom, left in face_locations:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    show_image(img)