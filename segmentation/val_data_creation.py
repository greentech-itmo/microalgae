import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
Script for preparing marked features/target data from .jpg to npy matrices
"""


for i in range(1, 10):
    img = cv2.imread(f'../photo_data/marked/{i}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255

    target = cv2.imread(f'../photo_data/marked/{i}target.jpg')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) / 255

    target[target != 0] = 1

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    c = axs[0].imshow(img, cmap='Grays')
    axs[0].set_title('Img')
    plt.colorbar(c, ax=axs[0])
    c = axs[1].imshow(target, cmap='Grays')
    axs[1].set_title('Target')
    plt.colorbar(c, ax=axs[1])
    plt.tight_layout()
    plt.show()

    np.save(f'../photo_data/marked_npy/img{i}.npy', img)
    np.save(f'../photo_data/marked_npy/target{i}.npy', target)