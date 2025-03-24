import os
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('D:/microalgae_data/target_concentration.csv')
conc_per_sample = []

predicted = []
targets = []

path = 'D:/microalgae_data'
for index, row in df.iterrows():
    cells_df = pd.read_csv(f'{path}/segmentation/cells.csv')
    name = row['name']
    cells = cells_df[name]
    print(f'Process {name}')
    dil_coef = row['dilution']
    target = row['concentration']
