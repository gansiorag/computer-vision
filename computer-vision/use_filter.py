"""
This module shows you how various filters work when applied to an image.

author Gansior A.G. gansior@gansior.ru
"""

import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt

"""
Фильтры
Предварительное преобразование по заранее определенным ядрам свертки (обычно 3x3). Например, фильтр четкости:
-1 	-1 	-1
-1 	 9 	-1
-1 	-1 	-1
Сумма всех коэффициентов в фильтре равна 1.
"""

def show_image (image):
    plt.figure(figsize=(16,8))
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("barcode.example.png")
    show_image(image)
    show_image(cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])))

    """
        Размытие изображения
        Обычный фильтр размытия:
        1/9 	1/9 	1/9
        1/9 	1/9 	1/9
        1/9 	1/9 	1/9
        Фильтр Гаусса:
        1/16 	2/16 	1/16
        2/16 	4/16 	2/16
        1/16 	2/16 	1/16
    """

    show_image(cv2.filter2D(image, -1, np.ones((3, 3)) / 9))
    show_image(cv2.filter2D(image, -1, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16))
    show_image(cv2.GaussianBlur(image, (3, 3), 0))
    show_image(cv2.medianBlur(image, 3))

    """
    Наращивание и эрозия
        Морфологические изменения (не "маски"). Наращивание берет максимальный элемент из заданной формы (окрестности), эрозия - минимальный. Форма крест/круг:
        0 	1 	0
        1 	1 	1
        0 	1 	0
        Форма квадрат (прямоугольник):
        1 	1 	1
        1 	1 	1
        1 	1 	1
         Открытие = наращивание + эрозия. Закрытие = эрозия + наращивание
    """

    kernel = np.ones((3, 3), np.uint8)
    show_image(cv2.erode(image, kernel, iterations=3))
    show_image(cv2.dilate(image, kernel, iterations=3))
    show_image(cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3))
    show_image(cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3))
