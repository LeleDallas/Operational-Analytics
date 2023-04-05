import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import sem, t


def ttest(data1, data2, alpha):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    se1, se2 = sem(data1),  sem(data2)
    sed = np.sqrt(se1**2.0 + se2**2.0)
    t_stat = (mean1-mean2)/sed
    degf = len(data1) + len(data2) - 2
    cv = t.ppf(1.0-alpha, degf)
    p = (1.0 - t.cdf(abs(t_stat), degf))*2.0
    return t_stat, degf, cv, p



df = pd.read_csv(
    'https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/Dataset.csv', header=0)
print(df)
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
pcc = np.corrcoef(df["ApplicantIncome"], df["LoanAmount"])
print(f"r={pcc}")
plt.plot(df["LoanAmount"], df["LoanAmount"], marker=".", linewidth=0)
plt.show()



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

#last part
data1 = df.loc[df["Gender"] == 'Male', 'ApplicantIncome']
data2 = df.loc[df["Gender"] == 'Female', 'ApplicantIncome']
t, degf, cv, p = ttest(data1, data2, 0.05)

print(t, degf, cv, p)
