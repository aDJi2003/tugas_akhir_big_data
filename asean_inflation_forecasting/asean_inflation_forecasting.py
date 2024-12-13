import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path = 'asean_inflation_data.xlsx'
raw_df = pd.read_excel(file_path)

data = raw_df.melt(id_vars=['Country'], var_name='Quarter', value_name='Value')

data['Year'] = data['Quarter'].str[:4].astype(int)
data['Quarter'] = data['Quarter'].str[4:]

def quarter_to_month(quarter):
    quarter_map = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
    return quarter_map.get(quarter, '01')

data['Month'] = data['Quarter'].apply(quarter_to_month)

data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'], format='%Y-%m')

data = data.set_index('Date')

ts_data = data[['Country', 'Value']]

asean_avg = ts_data.groupby(ts_data.index)['Value'].mean()

asean_avg.index = asean_avg.index.to_period('Q').to_timestamp()

decomposition = seasonal_decompose(asean_avg, model='additive', period=4)
decomposition.plot()
plt.show()

train = asean_avg[:-12]
test = asean_avg[-12:]

model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=4).fit()
forecast_steps = len(test) 
forecast = model.forecast(steps=forecast_steps)

if forecast.isnull().any():
    print("Warning: Forecast contains NaN values. Check input data or model settings.")
else:
    plt.figure(figsize=(10, 6))
    plt.plot(asean_avg.index, asean_avg, label="Actual")
    plt.plot(forecast.index, forecast, label="Forecast", linestyle="--")
    plt.title("Forecast vs Actual for ASEAN")
    plt.legend()
    plt.show()

    print("\nForecasted Values:")
    print(forecast)

    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mse)

    print(f"\nPerformance for ASEAN:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    asean_summary = pd.DataFrame({
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse]
    }, index=["ASEAN"])

    print("\nSummary of Forecasting Performance for ASEAN:")
    print(asean_summary)
