"""
This module make Локальные бинарные шаблоны

Используем LBPH (Local Binary Patterns Histogram) для обнаружения контуров лиц

Изображение: https://video.ittensive.com/machine-vision/faces.example.png

Локальные бинарные шаблоны


Author Gansior A. mail - gansior@gansior.ru tel - +79173383804
"""

import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt
from skimage import feature

def show_image (i):
    plt.figure(figsize=(16,8))
    plt.imshow(i[...,::-1])
    plt.show()

   # Бинарные шаблоны
    #
    # Бинарный уровень выбирается в зависимости от центрального пиксела в блоке 3x3 Преобразование бинарных шаблонов

def get_pixel(img, center, x, y):
    new_value = 0
    if y >= 0 and y < img.shape[1] and x >= 0 and x < img.shape[0] and img[x][y] >= center:
        new_value = 1
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
      64| 128 |   1
    ----------------
      32|   0 |   2
    ----------------
      16|   8 |   4
    '''
    center = img[x][y]
    val = 0
    val +=   1 * get_pixel(img, center, x-1, y+1) # top_right
    val +=   2 * get_pixel(img, center, x, y+1)   # right
    val +=   4 * get_pixel(img, center, x+1, y+1) # bottom_right
    val +=   8 * get_pixel(img, center, x+1, y)   # bottom
    val +=  16 * get_pixel(img, center, x+1, y-1) # bottom_left
    val +=  32 * get_pixel(img, center, x, y-1)   # left
    val +=  64 * get_pixel(img, center, x-1, y-1) # top_left
    val += 128 * get_pixel(img, center, x-1, y)   # top
    return val


if __name__ == '__main__':
    image = cv2.imread('faces.example.png')

    # Загрузка модели
    #
    # Используем обученную LBPH модель из пакета OpenCV
    # https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface.xml
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    image = cv2.imread('faces.example.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинарные шаблоны
    #
    # Бинарный уровень выбирается в зависимости от центрального пиксела в блоке 3x3 Преобразование бинарных шаблонов

    image_lbph = np.zeros(image.shape[:2])
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            image_lbph[i][j] = lbp_calculated_pixel(image_gray, i, j)
    show_image(image_lbph)

    # Обход пикселей начинается слева
    image_lbph = feature.local_binary_pattern(image_gray, 4, 1)
    show_image(image_lbph)

    # Обработка изображения
    #
    # Обнаружение лиц по каскадам LBPH
    faces = face_cascade.detectMultiScale(image_gray, 1.2, 8)

    # Отображение результата
    #
    # Выведем контуры лиц

    img = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    show_image(img)




