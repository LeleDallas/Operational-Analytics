import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


def gaussianChart(x, std, mean):
    x = (x - mean) / std
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / std


fig, axs = plt.subplots(2, 3)
fig.tight_layout(pad=2.0)
fig.set_figheight(6)
fig.set_figwidth(10)

df = pd.read_csv('../src/traffico16.csv')
npa = df['ago1'].to_numpy()

axs[0][0].hist(npa, bins=10, color='#00AA00',
               edgecolor='black', label="histogram")
axs[0][0].axis()

res = stats.relfreq(npa, numbins=10)

# print(res[0])
# print(df.iloc[:].describe())

axs[0][1].scatter(df.index, df['ott1'], label="Scatter")
axs[0][2].scatter(df['ago1'].sort_values(), df['ago2'].sort_values())


traffic_mean = npa.mean()


x = np.arange(-5, 5, 0.001)
axs[1][0].set_title("My Function")
axs[1][0].plot(gaussianChart(x, 1, np.mean(x)), label="std 1")
axs[1][0].plot(gaussianChart(x, 0.5, np.mean(x)), label="std 0.5")
axs[1][0].plot(gaussianChart(x, 2, np.mean(x)), label="std 2")

axs[1][1].plot(x, norm.pdf(x, np.mean(x), 1), label="std=1")
axs[1][1].plot(x, norm.pdf(x, np.mean(x), 0.5), label="std=0.5")
axs[1][1].plot(x, norm.pdf(x, np.mean(x), 2), label="std=2")

axs[1][2].plot(x, norm.cdf(x, np.mean(x), 1), label="std=1")
axs[1][2].plot(x, norm.cdf(x, np.mean(x), 0.5), label="std=0.5")
axs[1][2].plot(x, norm.cdf(x, np.mean(x), 2), label="std=2")


axs[0][0].legend()
axs[0][1].legend()
axs[1][0].legend()
axs[1][1].legend()
plt.legend()
plt.show()
