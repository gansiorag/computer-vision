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
    image = cv2.imread('faces.example.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.figure(figsize=(16, 8))
    plt.imshow(image[..., ::-1])
    plt.show()
