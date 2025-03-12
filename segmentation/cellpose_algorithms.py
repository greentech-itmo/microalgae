import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread

def segment_cells(image_path, diameter=None, channels=[0, 0], save_path=None):

    model = models.Cellpose(model_type='cyto3')
    img = imread(image_path)
    masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels)

    num_cells = len(np.unique(masks)) - 1  

    if save_path:
        plt.imsave(save_path, masks, cmap='gray')

    return masks, num_cells

def calculate_cell_area(mask):

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  

    if len(unique_labels) == 0:
        return 0

    total_area = np.sum(mask != 0)  
    return total_area / len(unique_labels)  

masks, num_cells = segment_cells("data/raw/BG-11-1-25-02-25-20-07-27.png")
average_area = calculate_cell_area(masks)
print(f"Найдено клеток: {num_cells}, Средняя площадь клетки: {average_area:.2f} пикселей")