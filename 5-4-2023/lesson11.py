import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


ds = pd.read_csv("../src/FilRouge.csv")
sarima_model = SARIMAX(ds["sales"], order=(
    0, 2, 2), seasonal_order=(0, 1, 0, 4))
sfit = sarima_model.fit()
sfit.plot_diagnostics(figsize=(10, 6))
plt.show()
