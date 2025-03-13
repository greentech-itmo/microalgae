import celldetection as cd
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from cv_algorithms import detect_cv2_objs

def detect_cell_celldetection(image_path):
    """
    Выполняет сегментацию клеток на изображении с помощью celldetection.
    """
    # Загружаем изображение
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Создаём модель celldetection
    model = cd.models.CellDetector()  # Использует предобученную модель

    # Выполняем детекцию
    instances = model.detect_instances(img)
    
    # Создаём бинарную маску
    mask = np.zeros_like(img)
    for instance in instances:
        mask[instance.mask] = 255  # Заполняем найденные области

    return mask

def validate_celldetection_with_real(save_path):
    if not os.path.exists(save_path):
        os.makedirs(f'{save_path}/images')

    df = pd.DataFrame(columns=['file_name',
                               'cells_num',
                               'AE',
                               'NAE',
                               'IOU',
                               'IOU(cells)',
                               'Cells area',
                               'Cells area error %',
                               'Cells area error pixels'])

    names, cells_num, aes, naes, ious, ious_one, areas, areas_errors, areas_errors_pixels = [], [], [], [], [], [], [], [], []

    # Отключаем интерактивный режим
    plt.ioff()

    for file in os.listdir(masks_path):
        mask = detect_cell_celldetection(f'{images_path}/{file}')
        img = cv2.imread(f'{masks_path}/{file}', cv2.IMREAD_GRAYSCALE)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].imshow(mask, cmap='gray')
        axs[0, 0].set_title('Detected')
        axs[0, 1].imshow(img, cmap='gray')
        axs[0, 1].set_title('Real')

        mask_g = np.where(mask > 0, 255, 0)
        img_g = np.where(img > 0, 255, 0)

        mask_objs = detect_cv2_objs(mask_g)
        real_objs = detect_cv2_objs(img_g)

        ae_cells = abs(real_objs.shape[0] - mask_objs.shape[0])
        nae_cells = abs((real_objs.shape[0] - mask_objs.shape[0]) / real_objs.shape[0])
        iou_val = iou(mask_g, img_g)
        iou_one_val = iou_one_class(mask_g, img_g)
        area_val = area_error(mask_g, img_g)
        area_pixel_val = area_error(mask_g, img_g, percent=False)

        axs[1, 0].imshow(mask, cmap='gray')
        for p in mask_objs:
            axs[1, 0].scatter(p[0], p[1], c='r', s=5)
        axs[1, 0].set_title('Detected')

        axs[1, 1].imshow(img, cmap='gray')
        for p in real_objs:
            axs[1, 1].scatter(p[0], p[1], c='r', s=5)
        axs[1, 1].set_title('Real')

        fig.suptitle(f'abs err={ae_cells}, norm abs err={nae_cells:.3f}, IOU={iou_val}, IOU(cells)={iou_one_val}, cells area error={area_val}')
        fig.tight_layout()
        fig.savefig(f'{save_path}/images/{file.split(".")[0]}_val.png')
        plt.close(fig)

        names.append(file.split(".")[0])
        cells_num.append(real_objs.shape[0])
        aes.append(ae_cells)
        naes.append(nae_cells)
        ious.append(iou_val)
        ious_one.append(iou_one_val)
        areas.append(np.sum(img_g == 0))
        areas_errors.append(area_val)
        areas_errors_pixels.append(area_pixel_val)

    df['file_name'] = names
    df['cells_num'] = cells_num
    df['AE'] = aes
    df['NAE'] = naes
    df['IOU'] = ious
    df['IOU(cells)'] = ious_one
    df['Cells area'] = areas
    df['Cells area error %'] = areas_errors
    df['Cells area error pixels'] = areas_errors_pixels

    df.to_csv(f'{save_path}/metrics.csv', index=False)

# Пути к данным
images_path = 'C:/Projects/microalgae/segmentation/data/raw'
masks_path = 'C:/Projects/microalgae/segmentation/data/marked'
out_folder = 'C:/Projects/microalgae/segmentation/data/validation_celldetection'

validate_celldetection_with_real(out_folder)
