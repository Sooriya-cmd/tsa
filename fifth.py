import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error
data=pd.read_csv(r"C:\Users\Admin\Downloads\5th_Sales.csv")
data["Date"]=pd.to_datetime(data["Date"],format="%d-%m-%Y")
data.set_index("Date",inplace=True)
series=data["Value"].dropna()
plt.figure(figsize=(10,5))
plt.plot(series)
plt.title("Original Time Series")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid(True)
plt.show()
def adf_test(series):
    result=adfuller(series)
    print("ADF Statistic:",result[0])
    print("p-value:",result[1])
    return result[1]<=0.05
stationary=adf_test(series)
d=0
stationary_series=series.copy()
while not stationary and d<2:
    d+=1
    stationary_series=series.diff(d).dropna()
    stationary=adf_test(stationary_series)
print("Selected d:",d)
lags=min(5,(len(stationary_series)-1)//2)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(stationary_series,lags=lags,ax=plt.gca())
plt.title("ACF")
plt.subplot(1,2,2)
plot_pacf(stationary_series,lags=lags,ax=plt.gca(),method="ywm")
plt.title("PACF")
plt.tight_layout()
plt.show()
best_aic=np.inf
best_order=None
best_model=None
for p in range(3):
    for q in range(3):
        try:
            model=ARIMA(series,order=(p,d,q))
            fitted_model=model.fit()
            if fitted_model.aic<best_aic:
                best_aic=fitted_model.aic
                best_order=(p,d,q)
                best_model=fitted_model
        except:
            pass
print("Best ARIMA Order:",best_order)
print(best_model.summary())
train_size=int(len(series)*0.8)
train=series.iloc[:train_size]
test=series.iloc[train_size:]
model=ARIMA(train,order=best_order)
model_fit=model.fit()
forecast=model_fit.forecast(steps=len(test))
rmse=np.sqrt(mean_squared_error(test,forecast))
mae=mean_absolute_error(test,forecast)
print("RMSE:",rmse)
print("MAE:",mae)
plt.figure(figsize=(12,6))
plt.plot(train.index,train,label="Training Data")
plt.plot(test.index,test,label="Actual")
plt.plot(test.index,forecast,label="Predicted",color="red")
plt.title("Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()