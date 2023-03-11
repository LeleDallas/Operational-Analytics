import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url, usecols=["data", "nuovi_positivi"])
dfinit = df.iloc[:]
plt.figure(figsize=(8, 4))
print(dfinit)
plt.plot(dfinit.nuovi_positivi, label="nuovi positivi")
plt.legend()
plt.show()
