import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv('../src/BoxJenkins.csv', header=0)
X = ds.Passengers.values
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]

# train = np.diff(train, 1)
# test = np.diff(test, 1)

diff1 = np.diff(test, 1)
diff2 = np.diff(train, 1)

# train = train[1:] + diff2

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
# plt.plot(train)
plt.plot(test)
test = test[:-1] + diff1
plt.plot([None for i in range(1, 2)] + [x for x in test])
plt.show()
