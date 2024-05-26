import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# 입력 데이터 가져오기
text = np.loadtxt('data_perceptron.txt')

# 데이터 포인트와 레이블로 나누기
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

# 입력 데이터 그래프 그리기
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# 각 차원에 대한 최댓값, 최솟값 지정하기
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# 출력 계층에 있는 뉴런의 수
# 데이터를 두 개의 클래스로 분리했기에 하나의 비트만으로 결과 표현 가능
# 따라서 출력 계층은 하나의 뉴런으로 구성됨
num_output = labels.shape[1]

# 2개의 입력 뉴런으로 구성된 퍼셉트론 정의
# 입력 데이터가 2차원이기 때문
# 각 차원마다 하나의 뉴런 할당
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1, dim2], num_output)

# 학습 데이터로 퍼셉트론 학습 시키기
error_progress = perceptron.train(data, labels, epochs=100, show=20, lr=0.03)

# 학습 과정을 그래프로 표시
# 오차 값 기준
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()

# 첫 번째 화면은 입력 데이터 포인트를 보여줌

# 두 번째 화면은 오차 값을 이용한 학습 진행 상태 보여줌
# 4번째 학습 주기(epoch)가 끝나는 시점에 오차가 0으로 떨어짐
