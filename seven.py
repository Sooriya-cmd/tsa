import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
df = pd.read_csv(r"C:\Users\Admin\Downloads\unemployment.csv")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.set_index("Date", inplace=True)
df = df[[
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]]
df = df.sort_index().dropna()
for col in df.columns:
    result = adfuller(df[col])
    print(col)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
data = df.copy()
for col in data.columns:
    if adfuller(data[col])[1] > 0.05:
        data[col] = data[col].diff()
data = data.dropna()
train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]
model = VAR(train)
lag_order = model.select_order(maxlags=5)
print(lag_order.summary())
lag = lag_order.aic
if lag is None or lag < 1:
    lag = 1
fitted_model = model.fit(lag)
forecast = fitted_model.forecast(
    train.values[-fitted_model.k_ar:],
    steps=len(test)
)
forecast = pd.DataFrame(
    forecast,
    index=test.index,
    columns=test.columns
)
print("\nForecasted Values:")
print(forecast)
for col in test.columns:
    rmse = np.sqrt(
        mean_squared_error(test[col], forecast[col])
    )
    print("\n", col)
    print("RMSE:", rmse)
    plt.figure(figsize=(8, 4))
    plt.plot(test.index, test[col], label="Actual")
    plt.plot(test.index, forecast[col], label="Forecast")
    plt.title(col)
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.show()