import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
df =pd.read_csv("C:\\Users\\Admin\\Documents\\revenue_data.csv")
df["date"]=pd.to_datetime(df["date"])
df["revenue"]=pd.to_numeric(df["revenue"],errors="coerce")
df = df.dropna(subset=["revenue"])
df.set_index("date",inplace=True)
plt.figure(figsize=(10,5))
plt.plot(df["revenue"],color ="blue")
plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()
df["Month"]=df.index.month
monthly_avg = df.groupby("Month")["revenue"].mean()
plt.figure(figsize=(8,5))
plt.plot(monthly_avg,monthly_avg.values,marker ="o")
plt.title("Monthly Average Revenue")
plt.xlabel("Month")
plt.ylabel("Average Revenue")
plt.xticks(range(1,13))
plt.grid(True)
plt.show()
result = seasonal_decompose(df["revenue"],model="additive",period=12)
result.plot()
plt.show()

