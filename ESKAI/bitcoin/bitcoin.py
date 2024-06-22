'''
비트코인 시세 예측하기
'''

# step.1 탐색: 시간 정보가 포함된 데이터 살펴보기
import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 출처 : # Data Source : https://www.blockchain.com/ko/charts/market-price?timespan=60days
file_path = '/Users/eunseo-ko/ECC/AIStudy/ESKAI/bitcoin/market-price.csv'
bitcoin_df = pd.read_csv(file_path, names = ['day','price'])

print(bitcoin_df.shape)
print(bitcoin_df.info())
bitcoin_df.tail()

bitcoin_df['day'] = pd.to_datetime(bitcoin_df['day'])

bitcoin_df.index = bitcoin_df['day']
bitcoin_df.set_index('day', inplace=True)

bitcoin_df.plot()
plt.show()

# step.2 예측: 파이썬 라이브러리를 활용해 시세 예측하기
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

model = ARIMA(bitcoin_df.price.values, order=(2,1,2))
model_fit = model.fit(trend='c', full_output=True, disp=True)
print(model_fit.summary())

fig = model_fit.plot_predict()
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()

forecast_data = model_fit.forecast(steps=5)

test_file_path = '/home/jaeyoon89/python-data-analysis/data/market-price-test.csv'
bitcoin_test_df = pd.read_csv(test_file_path, names=['ds', 'y'])

pred_y = forecast_data[0].tolist()
test_y = bitcoin_test_df.y.values

pred_y_lower = []
pred_y_upper = []
for lower_upper in forecast_data[2]:
    lower = lower_upper[0]
    upper = lower_upper[1]
    pred_y_lower.append(lower)
    pred_y_upper.append(upper)

plt.plot(pred_y, color="gold")
plt.plot(pred_y_lower, color="red")
plt.plot(pred_y_upper, color="blue")
plt.plot(test_y, color="green")

plt.plot(pred_y, color="gold")
plt.plot(test_y, color="green")

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

rmse = sqrt(mean_squared_error(pred_y, test_y))
print(rmse)