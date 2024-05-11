'''
시계열 데이터 이용하기
- 팬더를 사용하면 조건들을 이용한 데이터 필터링, 시계열 변수 합산 등 수행 가능
- 이로써 앱 만들 때 새로운 모델 안 만들어도 기존 모델 활용 가능
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from timeseries import read_data 

# Input filename
input_file = 'data_2D.txt'

# Load data
x1 = read_data(input_file, 2)
x2 = read_data(input_file, 3)

# Create pandas dataframe for slicing
data = pd.DataFrame({'dim1': x1, 'dim2': x2})

# Plot data
start = '1968'
end = '1975'
data[start:end].plot()
plt.title('Data overlapped on top of each other')

# Filtering using conditions
# - 'dim1' is smaller than a certain threshold
# - 'dim2' is greater than a certain threshold
data[(data['dim1'] < 45) & (data['dim2'] > 30)].plot()
plt.title('dim1 < 45 and dim2 > 30')

# Adding two dataframes 
plt.figure()
diff = data[start:end]['dim1'] + data[start:end]['dim2']
diff.plot()
plt.title('Summation (dim1 + dim2)')

plt.show()