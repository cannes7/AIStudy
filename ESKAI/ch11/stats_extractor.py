'''
시계열 데이터에서 통계 추출하기
- 유의미한 정보 추출 위해 평균, 분산, 상관관계, 최댓값 등 통계 분석 필요
- 이동 통계 방법: 창(간격) 안에서 시계열에 따라 연속적으로 통계를 계산하는 방법
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from timeseries import read_data 

# Input filename
input_file = 'data_2D.txt'

# Load input data in time series format
x1 = read_data(input_file, 2)
x2 = read_data(input_file, 3)

# Create pandas dataframe for slicing
data = pd.DataFrame({'dim1': x1, 'dim2': x2})

# Extract max and min values
print('\nMaximum values for each dimension:')
print(data.max())
print('\nMinimum values for each dimension:')
print(data.min())

# Extract overall mean and row-wise mean values
print('\nOverall mean:')
print(data.mean())
print('\nRow-wise mean:')
print(data.mean(1)[:12])

# Plot the rolling mean using a window size of 24
data.rolling(center=False, window=24).mean().plot()
plt.title('Rolling mean')

# Extract correlation coefficients
print('\nCorrelation coefficients:\n', data.corr())

# Plot rolling correlation using a window size of 60
plt.figure()
plt.title('Rolling correlation')
data['dim1'].rolling(window=60).corr(other=data['dim2']).plot()

plt.show()