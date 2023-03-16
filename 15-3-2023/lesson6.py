import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

fig, axs = plt.subplots(2, 3)
fig.tight_layout(pad=2.0)
fig.set_figheight(6)
fig.set_figwidth(10)
fig.suptitle('Lesson 6 Lab py')

transformer = StandardScaler()
y_passengers = pd.read_csv('../src/BoxJenkins.csv').Passengers

log = np.log(y_passengers)
log_diff = np.diff(log)
reshape = np.array(log_diff).reshape(len(log_diff), 1)
log_diff_scaled = transformer.fit_transform(reshape)


axs[0][0].plot(log, label="Log")
axs[0][1].plot(log_diff,  label="Log_Diff")
axs[0][2].plot(log_diff_scaled, label="Log_Diff_Scaled")

diff = log.diff()
diff.iat[0] = np.log(y_passengers.iat[0])
invertLog = np.exp(diff.cumsum())
invertDiff = log_diff.cumsum()
inverted = transformer.inverse_transform(log_diff_scaled)
axs[1][0].plot(invertLog, label="Log_Inverted")
axs[1][1].plot(invertDiff,  label="Log_Diff_Inverted")
axs[1][2].plot(inverted, label="Log_Diff_Scaled_Inverted")


axs[0][0].legend()
axs[0][1].legend()
axs[0][2].legend()
axs[1][0].legend()
axs[1][1].legend()
axs[1][2].legend()
plt.show()
