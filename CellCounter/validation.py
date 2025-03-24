import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from CellCounter.Segmentator import detect_cells, visualize_circles
from CellCounter.Shell import calculate_cells_count

df = pd.read_csv('D:/microalgae_data/target_concentration.csv')
conc_per_sample = []



path = 'D:/microalgae_data'
for index, row in df.iterrows():
    cells_df = pd.DataFrame()
    name = row['name']
    print(f'Process {name}')
    dil_coef = row['dilution']
    target = row['concentration']
    folder = f'{path}/{name}'
    save_img_path = f'{folder}/segmented'
    if not os.path.exists(folder):
        print(folder)
    cells_num = []
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    for file in os.listdir(folder):
        print(f'Process {folder}/{file}')
        if file != 'segmented':
            img = cv2.imread(f'{folder}/{file}')
            cells = detect_cells(img)
            #visualize_circles(img, cells, f'{save_img_path}/seg_{file}')
            #visualize_circles(img, cells)
            cells_num.append(cells.shape[1])
    '''concentrations = []
    for n in cells_num:
        cells_count = calculate_cells_count(n, dil_coef)
        concentrations.append(cells_count)
    conc_per_sample.append(concentrations)'''
    cells_df[name] = cells_num
    cells_df.to_csv(f'{folder}/segmented/cells.csv', index=False)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
axs.boxplot(conc_per_sample)
axs.set_xticks([y+1 for y in range(len(conc_per_sample))],
                  labels=df['name'].tolist(), rotation=20)
axs.scatter([y+1 for y in range(len(conc_per_sample))], df['concentration'])
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
axs.boxplot(conc_per_sample)
axs.set_xticks([y+1 for y in range(len(conc_per_sample))],
                  labels=df['name'].tolist(), rotation=20)
plt.tight_layout()
plt.show()