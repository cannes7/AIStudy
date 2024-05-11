'''
은닉 마르코프 모델 사용해 데이터 생성하기
- 은닉 마르코프 모델(HMM, Hidden Markov Model)
    순차적 데이터 중에서 눈으로 관측할 수 없는 내재(은닉)된 상태를 가지는 데이터를 처리하기 위한 마르코프 모델
    cf) 내재(은닉)된 상태를 가지는 데이터: 문장에서의 각 단어의 품사 등
    순차적 데이터 분석에 유용
    시스템이 은닉 상태의 마르코프 프로세스를 따른다고 가정
    즉, 시스템이 현재 눈으로 관찰할 수 없는 특정한 상태에 있고 다음 순서가 되면 다른 상태로 이동한다는 것
    일련의 상태 전이 과정을 거쳐 결과 값을 생성함
    결과 값 - 관측 가능, 상태의 변화 과정 - 관측 불가능
    이러한 상태 전이를 추론하는 것이 주 목적
    cf) https://ko.wilkipedia.org/wiki/은닉_마르코프_모델
'''
'''
세일즈맨이 여행을 시작하는 도시와 특정 시점에 어떤 도시에 있을 확률 계산
<- 마르코프 체인 사용

1. 전이 행렬 정의
전이 행렬: 각 도시에서 다른 도시로 이동할 확률
전이 행렬:
London      Barcelona    NY
0.10        0.70         0.20    (출발지가 London일 때)
0.75        0.15         0.10    (출발지가 Barcelona일 때)
0.60        0.35         0.05    (출발지가 NY일 때)

2. 화요일에 시작하여 금요일에 특정 도시에 있을 확률 계산
 이를 위해 전이 행렬을 반복적으로 적용하여 세일즈맨이 각 도시에 있을 확률을 계산해야 함

- 처음에는 화요일에 출발하는 경우를 고려하여 세일즈맨이 각 도시에 있을 확률을 초기화함
초기 확률: [0.10, 0.70, 0.20]

- 그 다음 화요일부터 금요일까지의 이동을 고려하여 전이 행렬을 반복적으로 적용함
금요일에 있을 확률 = 초기 확률 * 전이 행렬 ^ 3
'''


import datetime

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from timeseries import read_data 

# Load input data
data = np.loadtxt('data_1D.txt', delimiter=',')

# Extract the data column (third column) for training 
X = np.column_stack([data[:, 2]])

# Create a Gaussian HMM 
num_components = 5
hmm = GaussianHMM(n_components=num_components, 
        covariance_type='diag', n_iter=1000)

# Train the HMM 
print('\nTraining the Hidden Markov Model...')
hmm.fit(X)

# Print HMM stats
print('\nMeans and variances:')
for i in range(hmm.n_components):
    print('\nHidden state', i+1)
    print('Mean =', round(hmm.means_[i][0], 2))
    print('Variance =', round(np.diag(hmm.covars_[i])[0], 2))

# Generate data using the HMM model
num_samples = 1200
generated_data, _ = hmm.sample(num_samples) 
plt.plot(np.arange(num_samples), generated_data[:, 0], c='black')
plt.title('Generated data')

plt.show()