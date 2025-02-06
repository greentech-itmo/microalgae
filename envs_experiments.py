import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('phases_data/microalgae_data_envs.csv')

envs = df['Среда'].unique()


colors = ['r', 'g', 'b', 'm', 'y', 'pink']

for col in df.columns:
    if col != 'Сутки' and col != 'Среда':

        for i, env in enumerate(envs):
            sub_df = df[df['Среда'] == env]
            sub_df = sub_df.drop(columns=['Среда'])
            #sub_df = sub_df.groupby('Сутки').mean()

            plt.scatter(sub_df['Сутки'], sub_df[col], s=2, c=colors[i])

            daily_df_mean = sub_df.groupby('Сутки').mean()
            daily_df_max = sub_df.groupby('Сутки').max()
            daily_df_min = sub_df.groupby('Сутки').min()
            daily_data = daily_df_mean[col]
            plt.plot(daily_data.index, daily_data, c=colors[i], label=env)
            plt.fill_between(daily_data.index, daily_df_min[col], daily_df_max[col], alpha=0.3, color=colors[i])

        plt.title(col)
        plt.legend()
        plt.show()



