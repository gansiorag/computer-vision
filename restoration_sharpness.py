"""
This module make

author Gansior A.G. gansior@gansior.ru
"""


import numpy as np
import cv2
from skimage import exposure, restoration, color
import os, os.path
import matplotlib.pyplot as plt



def show_images (src, dst):
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    for a in ax:
        a.axis('off')
    ax[0].imshow(src)
    ax[1].imshow(dst)
    plt.show()


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


if __name__ == '__main__':
    image = cv2.imread("barcode.example.png")

    """
    Фильтр резкости
    Преобразование по матрице
    -1 	-1 	-1
    -1 	9 	-1
    -1 	-1 	-1
    """
    show_images(image, cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])))

    """
    Адаптивный фильтр резкости
    Преобразование по матрице
    -1 	-2 	-1
    -2 	13 	-2
    -1 	-2 	-1
    """
    show_images(image, cv2.filter2D(image, -1, np.array([[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]])))

    # Адаптивное изменение гистограммы
    # CLAHE
    show_images(image, exposure.equalize_adapthist(image))

    """
    Восстановления по Ричардсону-Люси
    Преполагаем, что размытие по равномерной маске, и последовательно восстанавливаем исходное изображание
    """
    show_images(image, restoration.richardson_lucy(color.rgb2gray(image), np.ones((5, 5)) / 25, iterations=5))

    """
    Маска четкости
    Размоем исходное изображение, а затем вычтем его из исходного. Разница усилит границы
    """
    show_images(image, unsharp_mask(image, amount=0.5, threshold=128))

    """
    Устранение шума
    Удаление пикселей, которые существенно отличаются от остальных
    """
    show_images(image, cv2.filter2D(cv2.fastNlMeansDenoisingColored(image), -1,
                                    np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])))