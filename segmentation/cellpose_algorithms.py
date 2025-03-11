import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread

def segment_cells(image_path, diameter=None, channels=[0, 0], save_path=None):
    """
    Сегментирует клетки на изображении с помощью Cellpose и возвращает маску и количество клеток.
    
    :param image_path: путь к изображению
    :param diameter: средний диаметр клеток (если None, определяется автоматически)
    :param channels: каналы, используемые для сегментации (по умолчанию [0,0] для серого изображения)
    :param save_path: путь для сохранения маски (если None, не сохранять)
    :return: (маска сегментации, количество клеток)
    """
    # Загрузка модели Cellpose
    model = models.Cellpose(model_type='cyto3')

    # Загрузка изображения
    img = imread(image_path)

    # Сегментация изображения
    masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels)

    # Подсчет количества клеток (разных объектов)
    num_cells = len(np.unique(masks)) - 1  # исключаем фон (метка 0)

    # Если указан путь сохранения, сохраняем маску
    if save_path:
        plt.imsave(save_path, masks, cmap='gray')

    return masks, num_cells

def calculate_cell_area(mask):
    """
    Вычисляет среднюю площадь клетки на основе сегментационной маски.
    
    :param mask: бинарная маска сегментации
    :return: средняя площадь клетки (в пикселях)
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # исключаем фон

    if len(unique_labels) == 0:
        return 0

    total_area = np.sum(mask != 0)  # количество пикселей, занятых клетками
    return total_area / len(unique_labels)  # средняя площадь одной клетки

masks, num_cells = segment_cells("C:/Projects/microalgae/segmentation/data/raw/BG-11-1-25-02-25-20-07-27.png")
average_area = calculate_cell_area(masks)
print(f"Найдено клеток: {num_cells}, Средняя площадь клетки: {average_area:.2f} пикселей")