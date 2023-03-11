import numpy as np
import pandas as pd
from scipy import stats  # to be used later
import matplotlib.pyplot as plt
import os
df = pd.read_csv('../src/traffico16.csv')  # dataframe (series)
npa = df['ago1'].to_numpy()  # numpy array
# plt.hist(npa, bins=10, color='#00AA00', edgecolor='black')
# plt.title(df.columns[0])
# plt.xlabel('num')
# plt.ylabel('days')
# plt.show()
# Rel freq
# res = stats.relfreq(npa, numbins=10)
# print(res[0])

print(df.iloc[:].describe())
plt.scatter(df.index, df['ott1'])
# plt.scatter(df['ago1'].sort_values(), df['ago2'].sort_values())
plt.show()
