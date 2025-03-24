from copy import deepcopy

import numpy as np
import cv2
from matplotlib import pyplot as plt



def shift_spectrum(image, channel_num:int):
    """
    Function for increasing one channel values
    :param image: cv2 object
    :param channel_num: int with number of channel for increasing values
    """
    chnls = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(chnls[channel_num])
    new_chs = []
    for i, c in enumerate(chnls):
        if i != channel_num:
            new_chs.append(c)
        else:
            new_chs.append(cl)
    img = cv2.merge(new_chs)
    return img

def visualize_circles(image, circles, save_path=None):
    """
    Function to save plots with detected objects
    :param image: cv2 object
    :param circles: list with x, y, R of detected objects
    """
    img_vis = deepcopy(image)
    plt.rcParams['figure.figsize'] = (8, 6)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img_vis, (i[0], i[1]), i[2], (255, 0, 0), 5)

    cells_area = np.round(np.sum(circles[0, :, 2] ** 2 * np.pi), 3)
    cells_mean_rad = int(np.round(np.mean(circles[0, :, 2])))
    plt.title(f'Full image cell number = {circles.shape[1]}\n'
              f'full cells area = {cells_area}\n'
              f'mean cell radius = {cells_mean_rad}')

    inds = np.arange(circles.shape[1])
    for ind in inds:
        plt.annotate(ind, (circles[0, ind, 0], circles[0, ind, 1]), c='blue', fontsize=9)
    plt.imshow(img_vis)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def return_mask(image, circles):
    """
    Function return image with black circles of objects on image
    """
    img_vis = deepcopy(image)
    for i in circles[0, :]:
        cv2.circle(img_vis, (i[0], i[1]), i[2], (0, 0, 0), -1)
    return img_vis

def detect_cells(image, increase_channel=None):
    """
    Function for cells detection in micro (small squares)
    :param increase_channel: int with number of channel for increasing values
    """
    img = image
    if increase_channel is not None:
        img = shift_spectrum(img, increase_channel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)

    minDist = 100
    param1 = 30
    param2 = 35  # smaller value-> more false circles
    minRadius = 15
    maxRadius = 100

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    if circles is None:
        print('Failed')
        return None

    circles = np.uint16(np.around(circles))
    return circles
