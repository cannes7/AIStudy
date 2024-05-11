'''
시계열 데이터 분할하기
- 데이터를 다양한 간격으로 나눈 후 필요한 정보 추출
- 분할 처리에서는 인덱스 대신 타임스탬프 사용함
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from timeseries import read_data

# Load input data
index = 2
data = read_data('data_2D.txt', index)

# Plot data with year-level granularity 
start = '2003'
end = '2011'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)

# Plot data with month-level granularity 
start = '1998-2'
end = '2006-7'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)

plt.show()