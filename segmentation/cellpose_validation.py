import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cellpose_algorithms import segment_cells, calculate_cell_area
from cellpose.io import imread
from skimage.io import imread
from cellpose import models
from skimage.color import rgb2gray

def iou(predicted, target):

    predicted = np.where(predicted != 0, 1, 0)
    target = np.where(target != 0, 1, 0)

    intersection = np.sum((predicted == 1) & (target == 1))
    union = np.sum((predicted == 1) | (target == 1))

    return np.round(intersection / union, 3) if union > 0 else 0

def area_error(predicted, target, percent=True):

    predicted = np.where(predicted != 0, 1, 0)
    target = np.where(target != 0, 1, 0)

    pred_area = np.sum(predicted)
    target_area = np.sum(target)

    if percent:
        return np.round(abs(pred_area - target_area) / target_area, 3) if target_area > 0 else 0
    return abs(pred_area - target_area)

def validate_segmentation(image_folder, mask_folder, save_path):

    if not os.path.exists(save_path):
        os.makedirs(f'{save_path}/images')

    results = []

    for file in os.listdir(mask_folder):
        image_path = os.path.join(image_folder, file)
        mask_path = os.path.join(mask_folder, file)

        if not os.path.exists(image_path):
            continue 

        predicted_mask, num_cells_pred = segment_cells(image_path)
        target_mask = imread(mask_path)

        if target_mask.ndim == 3:
            if target_mask.shape[2] == 4:  
                target_mask = target_mask[:, :, :3]  
            target_mask = rgb2gray(target_mask)
            target_mask = (target_mask * 255).astype(np.uint8) 

        num_cells_true = len(np.unique(target_mask)) - 1  

        ae_cells = abs(num_cells_true - num_cells_pred)
        nae_cells = abs(num_cells_true - num_cells_pred) / num_cells_true if num_cells_true > 0 else 0
        iou_value = iou(predicted_mask, target_mask)
        area_err = area_error(predicted_mask, target_mask)
        area_err_pixels = area_error(predicted_mask, target_mask, percent=False)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(predicted_mask, cmap='gray')
        axs[0].set_title('Segmented')
        axs[1].imshow(target_mask, cmap='gray')
        axs[1].set_title('Ground Truth')

        plt.suptitle(f'IoU={iou_value}, AE={ae_cells}, NAE={nae_cells:.3f}, Area Err={area_err}')
        
        images_path = os.path.join(save_path, "images")
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        plt.savefig(f'{images_path}/{file}_comparison.png')
        plt.close()

        
        results.append({
            'file_name': file,
            'true_cells_num': num_cells_true,
            'pred_cells_num': num_cells_pred,
            'AE': ae_cells,
            'NAE': nae_cells,
            'IOU': iou_value,
            'Area error %': area_err,
            'Area error pixels': area_err_pixels
        })

    df = pd.DataFrame(results)
    df.to_csv(f'{save_path}/metrics.csv', index=False)


validate_segmentation("data/raw", "data/marked", "data/cellpose_result")

