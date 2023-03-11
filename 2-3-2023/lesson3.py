import matplotlib.pyplot as plt
import pandas as pd

dfBox = pd.read_csv("../src/ETH-USD.csv")
x = dfBox.iloc[:].Date
y = dfBox.iloc[:]["Adj Close"]
plt.plot(x, y, label="ETH-USD Adj")
plt.legend()
plt.show()
