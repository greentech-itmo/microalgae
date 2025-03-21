import os

from segmentation.cv_algorithms import detect_cv2_objs

folder_path = 'D:/microalgae_data'
for samp_name in os.listdir(folder_path):
    if '.zip' not in samp_name:
        for img in os.listdir(samp_name):
            cells_objs = detect_cv2_objs(img)