from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = (10.0, 6.0)
df = pd.read_csv('../src/BoxJenkins.csv',usecols=["Passengers"])
ds = df[df.columns[0]] # converts to series
result = seasonal_decompose(ds, model='multiplicative',period=12)
result.plot()
plt.show()