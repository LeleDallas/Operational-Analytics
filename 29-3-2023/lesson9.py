import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import kstest

# data = np.array([-4, -3, 0.8, 1.8, 3.9, 6.2, 6.5])
# data = np.random.seed()

# np.random.seed(20)
# data = 20 * np.random.rand(100) + 100
# alpha = 0.05
# stat, p = shapiro(data)
# print('Shapiro=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
# if p > alpha:
#     print('Gaussian Sample, no rejection')
# else:
#     print('No Gaussian Sample no rejection ')
# stat, p = kstest(data, 'norm')
# print('Kolmogorov=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
# if p > alpha:
#     print('Gaussian sample, no rejection')
# else:
#     print('No Gaussian sample, rejection')

# qqplot(data, line='q')
# plt.show()

df = pd.read_csv(
    'https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/Dataset.csv', header=0)

data = df['ApplicantIncome'].to_numpy()
data = data[data < 9000]

alpha = 0.05
stat, p = shapiro(data)
print('Shapiro=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
if p > alpha:
    print('Gaussian Sample, no rejection')
else:
    print('No Gaussian Sample no rejection ')
stat, p = kstest(data, 'norm')
print('Kolmogorov=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
if p > alpha:
    print('Gaussian sample, no rejection')
else:
    print('No Gaussian sample, rejection')

qqplot(data, line='q')
plt.show()
