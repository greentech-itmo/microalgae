from cellpose import models
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from cv_algorithms import detect_cv2_objs

def iou(predicted, target):
    """
    Return intersection over union metric for two classes (classic)
    """
    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1
    diff = predicted - target
    tp = np.sum(diff == 0)
    fp = np.sum(diff == 1)
    fn = np.sum(diff == -1)
    return np.round(tp / (tp + fp + fn), 3)

def iou_one_class(predicted, target):
    """
        Return intersection over union metric for one class (only cells)
    """
    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1

    diff = predicted - target
    tp = np.sum(diff == 0)
    fp = np.sum(diff == 1)
    return np.round(tp / (tp + fp), 3)

def area_error(predicted, target, percent=True):
    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1

    if percent:
        return np.round(abs(np.sum(predicted) - np.sum(target))/np.sum(target), 3)
    else:
        return abs(np.sum(predicted) - np.sum(target))

def detect_cell_cellpose(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    model = models.Cellpose(model_type='cyto3')  

    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])

    mask = np.zeros_like(img)
    mask[masks > 0] = 255

    return mask

def validate_cellpose_with_real(save_path):
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

    names = []
    cells_num = []
    aes = []
    naes = []
    ious = []
    ious_one = []
    areas = []
    areas_errors = []
    areas_errors_pixels = []

    plt.ioff()

    for file in os.listdir(masks_path):
        mask = detect_cell_cellpose(f'{images_path}/{file}')
        img = cv2.imread(f'{masks_path}/{file}')

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].imshow(mask, cmap='gray')
        axs[0, 0].set_title('Detected')
        axs[0, 1].imshow(img, cmap='gray')
        axs[0, 1].set_title('Real')

        mask_g = np.mean(mask, axis=2) if mask.ndim == 3 else mask
        mask_g[mask_g != 0] = 255
        img_g = np.mean(img, axis=2) if img.ndim == 3 else img
        img_g[img_g != 0] = 255

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

        fig.suptitle(f'Abs err={ae_cells}, IOU={iou_val}, IOU(cells)={iou_one_val}, cells area error={area_val}')
        
        fig.tight_layout()
        fig.savefig(f'{save_path}/images/{file.split(".")[0]}_val.png')
        plt.close(fig)  # Закрываем именно fig, чтобы избежать зависания

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



images_path = 'C:/Projects/microalgae/segmentation/data/raw'
masks_path = 'C:/Projects/microalgae/segmentation/data/marked'
out_folder = 'C:/Projects/microalgae/segmentation/data/validation_cellpose'
validate_cellpose_with_real(out_folder)
# detect_cell_cellpose('C:/Projects/microalgae/segmentation/data/raw/BG-11-1-25-02-25-20-07-27.png')