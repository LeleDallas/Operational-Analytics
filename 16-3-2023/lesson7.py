import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# fig, axs = plt.subplots(3, 2)
# fig.tight_layout(pad=2.0)
# fig.set_figheight(6)
# fig.set_figwidth(10)
# fig.suptitle('Lesson 7 Lab py')


# ds = pd.read_csv('../src/BoxJenkinsNoData.csv').Passengers
# ds1 = ds.fillna(method="backfill")
# ds2 = ds.fillna(method="bfill")
# ds3 = ds.fillna(ds.mean)
# ds4 = ds.interpolate()


# axs[0][0].plot(ds1, label="Backfill")
# axs[0][1].plot(ds2,  label="Bfill")
# # axs[1][0].plot(ds3, label="Mean")
# axs[1][1].plot(ds4, label="Interpolate")
# axs[1][1].plot(ds4, label="Interpolate")
# axs[2][1].plot(ds, label="Interpolate")

# axs[0][0].legend()
# axs[0][1].legend()
# axs[1][0].legend()
# axs[2][1].legend()

# plt.show()


df=pd.read_csv("../src/traffico16.csv")
col = "ago2"
series = df[col].fillna(value=df[col].mean())
result = seasonal_decompose(series, model='additive',period=7)
observed=result.observed
trend = result.trend
seasonal=result.seasonal
resid=result.resid
std = resid.std()
plt.figure(figsize=(10,5))
plt.plot(resid,"o",label="datapoints")
plt.hlines(0,0,len(resid))
plt.hlines(1.5*std,0,len(resid),color="red",label="std limits")
plt.hlines(-1.5*std,0,len(resid),color="red")
plt.legend()
plt.show()