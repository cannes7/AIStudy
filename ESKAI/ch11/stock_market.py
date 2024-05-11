'''
주식 시장 분석하기
    주식 시장 데이터를 은닉 마르코프 모델을 사용해 분석
    주식 데이터는 타임스탬프가 붙어 있는 대표적인 시계열 데이터 중 하나
'''
'''
cf) matplotlib의 matplotlib.finance 모듈은 더 이상 사용되지 않아서
    대신 mplfinance 라이브러리 사용함
    pip install yfinance 터미널에 입력해야 함
'''

import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# Load historical stock quotes from Yahoo Finance
start = datetime.datetime(1971, 8, 30)
end = datetime.datetime(2015, 7, 7)
stock_data = yf.download('INTC', start=start, end=end)

# Extract the closing prices
closing_quotes = stock_data['Close'].values

# Extract the volume of shares traded
volumes = stock_data['Volume'].values[1:]

# Take the percentage difference of closing stock prices
diff_percentages = 100.0 * np.diff(closing_quotes) / closing_quotes[:-1]

# Take the list of dates starting from the second value
dates = np.arange(len(diff_percentages))

# Stack the differences and volume values column-wise for training
training_data = np.column_stack([diff_percentages, volumes])

# Create and train Gaussian HMM
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hmm.fit(training_data)

# Generate data using the HMM model
num_samples = 300
samples, _ = hmm.sample(num_samples)

# Plot the difference percentages
plt.figure()
plt.title('Difference percentages')
plt.plot(np.arange(num_samples), samples[:, 0], c='black')

# Plot the volume of shares traded
plt.figure()
plt.title('Volume of shares')
plt.plot(np.arange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)

plt.show()
