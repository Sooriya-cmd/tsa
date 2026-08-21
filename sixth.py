import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv(r"C:\Users\Admin\Downloads\agriculture_nonseasonal.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.set_index("Date", inplace=True)
df = df.asfreq("MS")
plt.plot(df["crop_yield"])
plt.title("Crop Yield")
plt.show()
result = adfuller(df["crop_yield"].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
train_size = int(len(df) * 0.8)
train = df["crop_yield"].iloc[:train_size]
test = df["crop_yield"].iloc[train_size:]
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
fit = model.fit()
forecast = fit.forecast(steps=len(test))
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
plt.plot(train, label="Training")
plt.plot(test, label="Actual")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.title("SARIMA Forecast")
plt.show()
future = fit.forecast(steps=12)
plt.plot(df["crop_yield"], label="Original")
plt.plot(future, label="Future Forecast")
plt.legend()
plt.title("Future Crop Yield Forecast")
plt.show()