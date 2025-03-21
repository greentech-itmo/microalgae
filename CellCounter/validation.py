import os

import pandas as pd

from segmentation.cv_algorithms import detect_cv2_objs

df = pd.read_csv('D:/microalgae_data/target_concentration.csv')
df

folder_path = 'D:/microalgae_data'

