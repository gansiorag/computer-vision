"""
This module make Каскады Хаара

Используем обученные каскады Хаара для обнаружения лиц на фотографии

Примитивы Хаара

Author Gansior A. mail - gansior@gansior.ru tel - +79173383804
"""

import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt
import glob
from skimage import feature


if __name__ == '__main__':
    #Загрузка обученных каскадов

    # Фактически, решающих деревьев, основанных на расположении примитивов (линий, точек) в изображении.
    #
    # Каскады по методу Виолы-Джонса:
    #
    #     интегральное представление изображения по признакам Хаара,
    #     классификатор на основе адаптивного бустинга,
    #     комбинирование классификаторов в оптимальную каскадную структуру.

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    list_file_images = glob.glob('faces/*.jpg')
    print(len(list_file_images))

    model_lbp = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    model_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    prototxt_path = "/home/al/PycharmProjects/computer-vision/deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model_dnn = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    kol_faces = 0
    for file_image in list_file_images:
        image = cv2.imread('faces.example.png')
        h, w = image.shape[:2]
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_haar = model_haar.detectMultiScale(image_gray, 1.3, 5)
        faces_lbp = model_lbp.detectMultiScale(image_gray, 1.2, 8)
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), image.mean(axis=0).mean(axis=0))
        model_dnn.setInput(blob)
        faces_dnn = np.squeeze(model_dnn.forward())
        ensemble = int(round(len(faces_haar)/4 + len(faces_lbp)/4 + len(np.where(faces_dnn[2]>0.5))/2 + 0.01))
        kol_faces += ensemble
        if ensemble != 1:
            print(file_image, ensemble)
    print(kol_faces)