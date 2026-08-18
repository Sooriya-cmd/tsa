import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
data = pd.read_csv("C:\\Users\\Admin\\Documents\\data_sales.csv")
data["date"]= pd.to_datetime(data["date"])
data.set_index("date",inplace =True)
series = data["sales"]
plt.figure(figsize=(10,5))
plt.plot(series,label="Original Data")
plt.title("Original Time Series")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
train = series[:-5]
test = series[-5:]
model=SimpleExpSmoothing(train)
fit=model.fit(optimized=True)
forecast=fit.forecast(len(test))
plt.figure(figsize=(10,5))
plt.plot(train,label="Training Data",linestyle=':')
plt.plot(test,label="Actual Test Data")
plt.plot(fit.fittedvalues,label="Smoothed values")
plt.plot(forecast,label="Forecast",color="red")
plt.title("First-order Exponential Smoothing")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
mae=mean_absolute_error(test,forecast)
rmse=np.sqrt(mean_squared_error(test,forecast))
print("Mean Absolute Error(MAE):",mae)
print("Root Mean Squared Error(RMSE):",rmse)
print("Smoothing parameter (Alpha):",fit.model.params["smoothing_level"])
    

