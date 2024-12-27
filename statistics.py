import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('phases_data/full_data.csv', sep=',', decimal=',')


'''for col in df.columns:
    vals = []
    if col not in ['Эксперимент', 'Сутки']:
        for i in range(1, 6):
            sub_df = df[df['Эксперимент']==i]
            sub_df = sub_df.drop(columns=['Эксперимент'])
            sub_df = sub_df.groupby('Сутки').mean()
            #norm_data = sub_df[col]/sub_df[col].max()
            norm_data = sub_df[col]
            plt.plot(sub_df.index, norm_data, label=f'Exp {i}')
        plt.title(col)
        plt.legend()
        plt.show()'''

colors = ['r', 'g', 'b', 'm', 'y', 'pink']

for col in df.columns:
    vals = []
    if col not in ['Эксперимент', 'Сутки']:
        for i in range(1, 6):
            sub_df = df[df['Эксперимент']==i]
            sub_df = sub_df.drop(columns=['Эксперимент'])

            norm_data = sub_df[col]
            plt.scatter(sub_df['Сутки'], norm_data, s=2, c=colors[i])

            daily_df_mean = sub_df.groupby('Сутки').mean()
            daily_df_max = sub_df.groupby('Сутки').max()
            daily_df_min = sub_df.groupby('Сутки').min()
            daily_data = daily_df_mean[col]
            plt.plot(daily_data.index, daily_data, c=colors[i], label=f'Exp {i}')
            plt.fill_between(daily_data.index, daily_df_min[col], daily_df_max[col],  alpha=0.3, color=colors[i])

        plt.title(col)
        plt.legend()
        plt.xlabel('Сутки')
        plt.show()