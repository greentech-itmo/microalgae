import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/validation_cellpose/metrics.csv')
df = df.sort_values(['cells_num'])

plt.scatter(df['cells_num'], df['AE'])
plt.xlabel('Cells number')
plt.ylabel('MAE')
plt.xlim(10, 55)
plt.ylim(-1, 20)
plt.title('Metric dependence of cells number')
plt.show()

plt.scatter(df['cells_num'], df['IOU(cells)'])
plt.xlabel('Cells number')
plt.ylabel('IOU')
plt.xlim(10, 55)
plt.ylim(0.96, 1)
plt.title('Metric dependence of cells number')
plt.show()