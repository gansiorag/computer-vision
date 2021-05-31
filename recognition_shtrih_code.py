"""
This module make

author Gansior A.G. gansior@gansior.ru
"""
import pprint

from pyzbar.pyzbar import decode
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np


def show_image (image):
    plt.figure(figsize=(16,8))
    plt.imshow(np.uint8(np.absolute(image)))
    plt.show()



if __name__ == '__main__':
    list_files = glob.glob('barcodes/*.png')
    for image in list_files:
        imagess = cv2.imread(image)
        show_image(imagess)
        detectedBarcodes = decode(imagess)
        for barcode in detectedBarcodes:
            pprint.pprint(barcode)
