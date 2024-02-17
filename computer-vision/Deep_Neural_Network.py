"""
This module make Deep Neural Network

Используем обученную SSD модель глубокой нейронной сети (DNN) для обнаружения контуров лиц

Изображение: https://video.ittensive.com/machine-vision/faces.example.png

Глубокие нейросети в OpenCV


Author Gansior A. mail - gansior@gansior.ru tel - +79173383804
"""

import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt

def show_image (i):
    plt.figure(figsize=(16,8))
    plt.imshow(i[...,::-1])
    plt.show()

if __name__ == '__main__':
    image = cv2.imread('faces.example.png')
    # Загрузка модели
    #
    # Используем DNN из пакета OpenCV
    # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    prototxt_path = "/home/al/PycharmProjects/computer-vision/deploy.prototxt"
    # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
    model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    # Обработка изображения
    #
    # Исходный размер и вычисление средних уровней цвета

    h, w = image.shape[:2]
    avg = image.mean(axis=0).mean(axis=0)
    print(avg)
    #Изменение размера и нормализация цвета
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), avg)
    # Выделение контуров лиц
    model.setInput(blob)
    output = np.squeeze(model.forward())

    # Отображение результата
    #
    # Выведем контуры лиц и процент уверенности в лицах

    img = image.copy()
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        if confidence > 0.5:
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(np.int)
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
            cv2.putText(img, f"{confidence * 100:.2f}%",
                        (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    show_image(img)