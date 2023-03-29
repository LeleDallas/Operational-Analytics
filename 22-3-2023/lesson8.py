import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

detrended_series = signal.detrend(numbers, type="linear")
fft = np.fft.fft(detrended_series)
one_sided_fft = fft[:int(n/2)+1]
# magnitude is power of 2 divided by l
