import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

i = 1

df = pd.read_csv('phases_data/full_data.csv', sep=',', decimal=',')
sub_df = df[df['Эксперимент'] == i]
daily_df_mean = sub_df.groupby('Сутки').mean()


degree = 3

X = np.array(daily_df_mean.index)
y = np.array(daily_df_mean['Кол-во клеток, ед/мл суспензии.'])


coeffs = np.polyfit(X, y, degree)
poly = np.poly1d(coeffs)
y_plot = np.polyval(poly, X)

plt.scatter(daily_df_mean.index, daily_df_mean['Кол-во клеток, ед/мл суспензии.'], c='pink', label='Train points')
plt.plot(np.arange(y_plot.shape[0]), y_plot, c='black', label=f'Polynomial interpolation - {degree} deg\ncoeffs={np.round(coeffs, 3)}')
plt.title(f'Кол-во клеток, ед/мл суспензии. - Exp{i}')
plt.legend()
plt.show()